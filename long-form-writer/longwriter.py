import argparse
import google.generativeai as genai
import re
import time
import random
import json
import os
from datetime import datetime
import logging

def setup_logging(title):
    """Sets up logging configuration."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:30]
    log_filename = f'logs/{safe_title}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

def get_user_input():
    """Gets the title, desired length, genre, language, and optional context from the user."""
    parser = argparse.ArgumentParser(description="Generate long-form articles.")
    parser.add_argument("title", type=str, help="The title of the article.")
    parser.add_argument("length", type=int, help="The desired length of the article in words.")
    parser.add_argument("genre", type=str, help="The genre of the article (e.g., academic paper, science fiction).")
    parser.add_argument("language", type=str, help="The desired language (e.g., en, zh).")
    parser.add_argument("--context", type=str, default=None, help="Optional context information to guide text generation.")
    args = parser.parse_args()
    return args.title, args.length, args.genre, args.language, args.context

def parse_outline(outline_text):
    """Parses the API response into a structured outline."""
    logging.info(f"Parsing outline text:\n{outline_text}")
    try:
        # 提取JSON部分
        match = re.search(r"```json\n(.*?)```", outline_text, re.DOTALL)
        if match:
            json_string = match.group(1)
            outline = json.loads(json_string)
            
            # 验证outline格式
            if not isinstance(outline, dict) or "sections" not in outline:
                raise ValueError("Invalid outline format: missing 'sections' key")
            
            # 验证每个章节的必要字段
            for section in outline["sections"]:
                required_fields = ["title", "length", "level"]
                missing_fields = [field for field in required_fields if field not in section]
                if missing_fields:
                    raise ValueError(f"Missing required fields in section: {missing_fields}")
            
            logging.info(f"Successfully parsed outline: {json.dumps(outline, ensure_ascii=False, indent=2)}")
            return outline["sections"]
        else:
            raise ValueError("Could not find JSON data in API response")
    except Exception as e:
        logging.error(f"Error parsing outline: {e}")
        return []

def generate_outline(title, length, genre, language, context):
    """Generates a detailed outline for the article using the Gemini API."""
    logging.info(f"Generating outline for title: {title}, length: {length}, genre: {genre}, language: {language}")
    
    prompt = f"""作为一位专业的{genre}作家，请为题为「{title}」的{length}字{genre}创建详细的分层大纲。

要求：
1. 内容规划
- 确保内容结构完整，逻辑连贯
- 各章节主题明确，避免内容重复
- 合理分配篇幅，确保总字数接近{length}字
- 考虑读者需求，合理安排内容深度

2. 结构设计
- 至少包含三个层级的结构
- 每个章节和小节都应有清晰的主题焦点
- 确保章节之间的层级关系合理
- 避免结构过于零散或过于集中

3. 字数分配
- 各子节字数应与内容重要性相匹配

请以下列JSON格式输出大纲：
{{
    "sections": [
        {{
            "title": "节标题",
            "length": 预计字数,
            "level": 层级数字(1为顶级),
            "key_points": ["关键点1", "关键点2"]  // 该节需要涵盖的要点
        }}
    ]
}}"""

    if context:
        prompt += f"""

参考以下背景信息规划内容：
{context}

注意：
- 将上述背景信息合理融入相关章节
- 确保内容与背景信息保持一致
- 适当扩展或深化相关主题"""

    if language.lower() in ['zh', 'chinese', 'cn']:
        prompt += """

输出要求：
- 使用中文输出
- 确保专业术语准确
- 保持用语正式得体"""
    
    try:
        outline_text = call_gemini_api(prompt)
        outline = parse_outline(outline_text)
        if not outline:
            raise ValueError("Failed to generate valid outline")
        return outline
    except Exception as e:
        logging.error(f"Error in generate_outline: {e}")
        return []

def call_gemini_api(prompt, max_retries=10):
    """Calls the Gemini API to generate text, with retries."""
    genai.configure(api_key="AIzaSyCeV5Cu43yzOEkqFnHsU8Vi2RDDukYznWc")
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt + random.uniform(0, 2)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise

    raise Exception("Failed to call Gemini API after multiple retries.")

def extract_main_points(content):
    """提取已完成内容的主要论点"""
    points = []
    paragraphs = content.split('\n\n')
    for p in paragraphs:
        if len(p.strip()) > 100:
            points.append(p[:100] + "...")
    return "\n".join(f"- {p}" for p in points[:3])

def identify_transition_points(content, next_title):
    """识别需要与下一节建立连接的关键点"""
    last_paragraphs = content.split('\n\n')[-2:]
    return "\n".join(f"- {p[:100]}..." for p in last_paragraphs)

def write_section(title, length, genre, language, outline, written_content, parent_title=None, context=None):
    """Writes a single section of the article with enhanced context awareness."""
    # 获取未完成的大纲部分
    remaining_outline = get_remaining_outline(outline, written_content)
    
    # 转换剩余大纲为结构化文本
    outline_text = "待完成章节：\n"
    for item in remaining_outline:
        indent = "  " * (item["level"] - 1)
        outline_text += f"{indent}• {item['title']} ({item['length']}字)\n"
        
        # 修复 key_points 处理逻辑
        if "key_points" in item and isinstance(item["key_points"], list):
            # 确保 key_points 中的所有元素都是字符串
            key_points = [str(point) for point in item["key_points"]]
            outline_text += f"{indent}  要点：{', '.join(key_points)}\n"
        
        outline_text += "\n"
    
    # 构建上下文信息
    context_info = ""
    if written_content:
        context_info = f"""已完成内容：
{written_content}

写作进度分析：
1. 已完成的主要论点：
{extract_main_points(written_content)}
2. 需要加强的连接点：
{identify_transition_points(written_content, title)}
"""

    # 构建主提示词
    prompt = f"""作为一位专业的{genre}作家，请完成以下写作任务：

任务概述：
- 当前章节：{title}
- 目标字数：{length}字
- 文体类型：{genre}
- 写作语言：{language}
{f'- 所属章节：{parent_title}' if parent_title else ''}

写作背景：
{context_info}

剩余结构：
{outline_text}

写作要求：
1. 内容要求
- 紧扣当前章节主题「{title}」
- 确保与已完成内容保持连贯性
- 内容深度与字数要求({length}字)相匹配
- 注意过渡自然，逻辑清晰

2. 风格要求
- 保持{genre}的专业写作风格
- 用语精准，表达清晰
- 适当使用专业术语
- 注意语言的连贯性和流畅度

3. 结构要求
- 段落组织合理，层次分明
- 确保论述完整，不重复已写内容
- 为下一章节做好铺垫
- 注意段落之间的过渡自然

请直接输出正文内容，无需包含标题。"""

    if context:
        prompt += f"""

参考背景信息：
{context}

注意：
- 将背景信息自然融入内容
- 确保观点与背景信息协调
- 适当补充或深化相关论述"""

    logging.info(f"Writing section: {title}")
    logging.debug(f"Full prompt for section {title}:\n{prompt}")
    
    content = call_gemini_api(prompt)
    return content

def get_remaining_outline(outline, written_content):
    """Extract unwritten sections from the outline based on written content."""
    remaining = []
    written_titles = set()
    title_pattern = re.compile(r'^#{1,3}\s+(.+?)\s*$', re.MULTILINE)
    
    for match in title_pattern.finditer(written_content):
        written_titles.add(match.group(1).strip())
    
    for item in outline:
        if item["title"] not in written_titles:
            remaining.append(item)
    
    return remaining

def assemble_article(outline, title, length, genre, language, context=None):
    """Assembles the article while maintaining context awareness."""
    logging.info("Assembling article...")
    final_article = ""
    written_content = ""
    current_level_1_title = ""
    
    for item in outline:
        if written_content and f"# {item['title']}" in written_content:
            continue
            
        # Add section header based on level
        header = f"{'#' * item['level']} {item['title']}\n\n"
        if item["level"] == 1:
            current_level_1_title = item["title"]
        
        # Generate content with enhanced context
        content = write_section(
            item["title"], 
            item["length"], 
            genre, 
            language, 
            outline,
            written_content,
            parent_title=current_level_1_title if item["level"] > 1 else None,
            context=context
        )
        
        written_content += header + content + "\n\n"
        final_article += header + content + "\n\n"

    return final_article

def save_article(title, content):
    """Saves the article content to a markdown file."""
    if not os.path.exists('articles'):
        os.makedirs('articles')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:30]
    filename = f'articles/{safe_title}_{timestamp}.md'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logging.info(f"Article saved to: {filename}")
    return filename

def main():
    """Main function."""
    title, length, genre, language, context = get_user_input()
    
    # 设置日志
    log_file = setup_logging(title)
    logging.info(f"Starting article generation for: {title}")
    logging.info(f"Log file created at: {log_file}")
    
    # 生成大纲
    outline = generate_outline(title, length, genre, language, context)
    if not outline:
        logging.error("Exiting due to outline generation failure.")
        return

    # 保存大纲到 JSON 文件
    if not os.path.exists('articles'):
        os.makedirs('articles')

    # 先清理标题，避免在 f-string 中使用反斜杠
    cleaned_title = re.sub(r"[^\w\s-]", "", title).strip()[:30]
    outline_file = f'articles/{cleaned_title}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_outline.json'
    
    with open(outline_file, 'w', encoding='utf-8') as f:
        json.dump({"sections": outline}, f, ensure_ascii=False, indent=2)
    logging.info(f"Outline saved to: {outline_file}")

    # 生成文章，增强上下文感知能力
    final_article = assemble_article(outline, title, length, genre, language, context)
    
    # 保存最终文章
    article_file = save_article(title, final_article)
    
    logging.info("Article generation completed successfully")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Outline file: {outline_file}")
    logging.info(f"Article file: {article_file}")

if __name__ == "__main__":
    main()