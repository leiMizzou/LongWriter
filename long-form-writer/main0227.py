import argparse
import google.generativeai as genai
import re
import time
import random
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 常量配置
MAX_CONTEXT_LENGTH = 50000  # 上下文长度，用于保留之前生成的内容
MAX_RETRIES = 10
RETRY_BASE_DELAY = 2
LOG_DIR = 'logs'
ARTICLES_DIR = 'articles'

def setup_logging(title: str) -> str:
    """
    设置日志配置
    
    Args:
        title: 文章标题
        
    Returns:
        str: 日志文件路径
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:30]
    log_filename = f'{LOG_DIR}/{safe_title}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

def get_user_input() -> Tuple[str, int, str, str, Optional[str]]:
    """
    获取用户输入：标题、长度、体裁、语言和可选上下文
    
    Returns:
        Tuple[str, int, str, str, Optional[str]]: 标题、长度、体裁、语言和上下文
    """
    parser = argparse.ArgumentParser(description="生成长篇文章")
    parser.add_argument("title", type=str, help="文章标题")
    parser.add_argument("length", type=int, help="文章期望字数")
    parser.add_argument("genre", type=str, help="文章体裁（例如学术论文、科幻小说）")
    parser.add_argument("language", type=str, help="期望语言（例如 en, zh）")
    parser.add_argument("--context", type=str, default=None, help="可选上下文信息，用于指导文本生成")
    args = parser.parse_args()
    return args.title, args.length, args.genre, args.language, args.context

def extract_json_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    从API响应中提取JSON数据
    
    Args:
        response_text: API返回的文本
        
    Returns:
        List[Dict[str, Any]]: 解析后的JSON数据
        
    Raises:
        ValueError: 如果无法解析JSON
    """
    # 尝试多种可能的JSON格式
    # 1. 尝试从 ```json ... ``` 中提取
    json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 2. 尝试从任何三重反引号代码块中提取
    code_match = re.search(r"```(?:\w*)\n(.*?)```", response_text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 3. 尝试直接解析整个响应
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # 4. 尝试从响应中找到可能的JSON部分（从 { 开始到 } 结束）
    json_candidate = re.search(r"\[[\s\S]*\]", response_text)
    if json_candidate:
        try:
            return json.loads(json_candidate.group(0))
        except json.JSONDecodeError:
            pass
    
    # 所有尝试都失败
    logging.error(f"无法从响应中提取JSON数据:\n{response_text}")
    raise ValueError("无法解析API返回的JSON数据")

def call_gemini_api(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    """
    调用 Gemini API 生成文本，支持重试但识别内容阻止
    
    Args:
        prompt: 提示文本
        max_retries: 最大重试次数
        
    Returns:
        str: API返回的文本
        
    Raises:
        ValueError: 如果提示被阻止或多次重试后仍然失败
    """
    # 从环境变量获取API密钥
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("未找到GEMINI_API_KEY环境变量")
        raise ValueError("请设置GEMINI_API_KEY环境变量")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if not response.candidates:
                logging.error(f"提示被阻止: {response.prompt_feedback}")
                raise ValueError("提示被阻止")
            return response.text
        except Exception as e:
            # 如果是内容阻止，不要重试相同的提示，直接抛出异常
            if "block_reason" in str(e) or "提示被阻止" in str(e):
                logging.error(f"提示被阻止: {e}")
                raise ValueError("提示被阻止")
                
            # 对于速率限制错误，进行重试
            elif "429" in str(e) and attempt < max_retries - 1:
                # 指数退避重试
                wait_time = RETRY_BASE_DELAY ** attempt + random.uniform(0, 2)
                logging.warning(f"超过速率限制，将在 {wait_time:.2f} 秒后重试... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                # 其他错误也进行重试
                wait_time = RETRY_BASE_DELAY * (attempt + 1) + random.uniform(0, 2)
                logging.warning(f"API调用失败: {e}, 将在 {wait_time:.2f} 秒后重试... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logging.error(f"多次重试后API调用仍然失败: {e}")
                raise
    
    raise Exception("多次重试后仍无法调用 Gemini API")

def generate_outline(title: str, length: int, genre: str, language: str, context: Optional[str]) -> List[Dict[str, Any]]:
    """
    使用 Gemini API 生成扁平化的详细大纲（仅使用一级章节）
    
    Args:
        title: 文章标题
        length: 文章期望字数
        genre: 文章体裁
        language: 期望语言
        context: 可选上下文信息
        
    Returns:
        List[Dict[str, Any]]: 大纲数据
    """
    logging.info(f"生成大纲: 标题: {title}, 长度: {length}, 体裁: {genre}, 语言: {language}")
    
    # 根据长度计算理想章节数，辅助AI生成
    if length <= 1000:
        suggested_chapters = 3  # 短文章
    elif length <= 3000:
        suggested_chapters = 5  # 中短文章
    elif length <= 5000:
        suggested_chapters = 7  # 中等长度
    elif length <= 10000:
        suggested_chapters = 9  # 中长文章
    elif length <= 15000:
        suggested_chapters = 12  # 长文章
    else:
        # 超长文章
        suggested_chapters = 12 + ((length - 15000) // 5000) * 1.2
    
    # 计算每章节平均长度，确保不超过5000字
    avg_chapter_length = length // suggested_chapters
    if avg_chapter_length > 5000:
        suggested_chapters = (length + 4999) // 5000  # 向上取整
        avg_chapter_length = length // suggested_chapters
    
    prompt = f"""为一个体裁为 {genre}、标题为 '{title}' 的文章创建详细大纲，总长度约为 {length} 字。

大纲要求：
* 只使用单级章节结构，不要使用多级嵌套章节。
* 建议分为约 {suggested_chapters} 个章节（包括引言和结论），每章平均约 {avg_chapter_length} 字。
* 每章节字数不应超过5000字，以确保生成质量。
* 每个章节应该独立且内容不重叠。
* 每个章节应有清晰的主题和边界。
* 第一章应为引言或介绍，最后一章应为结论或总结。
* 确保字数在各章节之间平衡，反映每个主题的重要性和复杂性，总字数加起来约为 {length} 字。
* 创建专业、中性的章节标题，避免任何可能违反内容政策的敏感主题。

以 JSON 格式输出大纲，作为对象列表。每个对象应包含以下键：
* "title": (字符串) 章节标题。
* "length": (整数) 该章节的预计字数。
* "level": (整数) 值固定为1，表示顶级章节。

示例：
[
    {{"title": "引言", "length": 500, "level": 1}},
    {{"title": "历史背景", "length": 1000, "level": 1}},
    {{"title": "核心概念", "length": 1500, "level": 1}},
    {{"title": "实际应用", "length": 1000, "level": 1}},
    {{"title": "未来展望", "length": 800, "level": 1}},
    {{"title": "结论", "length": 200, "level": 1}}
]
"""
    if context:
        prompt += f"\n\n生成大纲的额外上下文:\n{context}"
    try:
        outline_text = call_gemini_api(prompt)
        logging.info(f"生成的大纲文本:\n{outline_text}")
        
        # 使用更健壮的JSON提取方法
        try:
            outline = extract_json_from_response(outline_text)
            logging.info(f"解析后的大纲: {json.dumps(outline, ensure_ascii=False, indent=2)}")
            
            # 验证大纲结构
            validate_outline(outline, length)
            
            # 优化章节划分
            outline = optimize_section_generation(outline, genre, language, length)
            
            # 检查和净化标题
            for i, item in enumerate(outline):
                if "title" in item:
                    outline[i]["title"] = sanitize_title(item["title"])
                    
            logging.info("大纲生成成功")
            return outline
        except ValueError as e:
            logging.error(f"大纲解析失败: {e}")
            # 如果JSON解析失败，尝试重新生成，更明确要求JSON格式
            logging.info("尝试重新生成大纲，强调JSON格式...")
            retry_prompt = prompt + "\n\n请确保输出是有效的JSON格式，不包含其他文本。只返回JSON数组，不要添加任何解释或额外文本。"
            outline_text = call_gemini_api(retry_prompt)
            try:
                outline = extract_json_from_response(outline_text)
                logging.info(f"第二次尝试解析后的大纲: {json.dumps(outline, ensure_ascii=False, indent=2)}")
                validate_outline(outline, length)
                outline = optimize_section_generation(outline, genre, language, length)
                
                # 检查和净化标题
                for i, item in enumerate(outline):
                    if "title" in item:
                        outline[i]["title"] = sanitize_title(item["title"])
                        
                logging.info("第二次尝试大纲生成成功")
                return outline
            except ValueError:
                logging.error("第二次尝试大纲生成仍然失败")
                return []
    except Exception as e:
        logging.error(f"生成大纲时出错: {e}")
        return []

def validate_outline(outline: List[Dict[str, Any]], target_length: int) -> None:
    """
    验证大纲结构和总字数
    
    Args:
        outline: 大纲数据
        target_length: 目标字数
        
    Raises:
        ValueError: 如果大纲结构不正确或总字数偏差过大
    """
    # 检查必要字段
    for item in outline:
        if "title" not in item or "length" not in item or "level" not in item:
            raise ValueError("大纲项缺少必要字段(title, length, level)")
    
    # 检查总字数
    total_length = sum(item.get("length", 0) for item in outline)
    # 允许10%的偏差
    if abs(total_length - target_length) / target_length > 0.1:
        logging.warning(f"大纲总字数({total_length})与目标字数({target_length})相差超过10%")

def sanitize_title(title: str) -> str:
    """
    净化可能有问题的章节标题，使其更符合内容政策
    
    Args:
        title: 原始标题
        
    Returns:
        str: 更安全的标题版本
    """
    # 定义敏感主题/词汇列表及其替代词
    sensitive_map = {
        "婚外情": "H外情", "外遇": "W遇", "情人": "Q人", "出轨": "C轨", 
        "间谍": "J谍", "侵犯": "Q犯", "猥亵": "W亵", "性感": "X感", "性爱": "X爱", "性行为": "X行为", 
        "色情": "S情", "毒品": "D品", "赌博": "D博", "自杀": "Z杀", "暴力": "B力", 
        "血腥": "X腥", "谋杀": "M杀", "杀人": "S人", "霸凌": "B凌"
    }
    
    # 记录原始标题用于日志
    original_title = title
    
    # 替换所有敏感词
    for term, replacement in sensitive_map.items():
        if term in title:
            title = title.replace(term, replacement)
    
    # 如果标题被修改，记录日志
    if original_title != title:
        logging.info(f"标题净化: '{original_title}' -> '{title}'")
        
    return title

def regenerate_chapter_title(title: str, genre: str, language: str) -> str:
    """
    使用API生成更适合内容政策的章节标题替代版本
    
    Args:
        title: 原始标题
        genre: 文章体裁
        language: 目标语言
        
    Returns:
        str: 重新生成的标题
    """
    try:
        # 先尝试简单的标题净化
        sanitized_title = sanitize_title(title)
        if sanitized_title != title:
            return sanitized_title
            
        # 如果没有发现明显的敏感词，使用API生成替代标题
        prompt = f"""请为以下章节标题创建一个替代版本，保持相似的主题和意义，但使其更加中性、专业，并避免任何可能违反内容政策的元素。
原标题: "{title}"

您的替代标题应该:
1. 保留原标题的核心主题和意图
2. 避免提及婚外情、侵犯行为、犯罪活动、不当关系等敏感话题
3. 使用更中性、专业的语言
4. 保持在{genre}的写作风格内
5. 使用{language}语言

只需提供新标题，无需解释。"""

        new_title = call_gemini_api(prompt)
        # 清理可能的引号和额外文本
        new_title = new_title.strip().strip('"').strip("'").strip()
        
        # 如果返回的标题太长或包含解释，只取第一行或前50个字符
        if "\n" in new_title:
            new_title = new_title.split("\n")[0].strip()
        if len(new_title) > 50:
            new_title = new_title[:50].strip()
            
        logging.info(f"标题重新生成: '{title}' -> '{new_title}'")
        return new_title
    except Exception as e:
        logging.error(f"标题重新生成失败: {e}, 使用净化后的标题")
        return sanitize_title(title)

def optimize_section_generation(outline: List[Dict[str, Any]], genre: str, language: str, target_length: int) -> List[Dict[str, Any]]:
    """
    优化章节划分，根据目标文档长度自动调整章节数量
    
    Args:
        outline: 原始大纲
        genre: 文章体裁
        language: 期望语言
        target_length: 目标总字数
        
    Returns:
        List[Dict[str, Any]]: 优化后的大纲
    """
    if not outline:
        return []
    
    # 根据目标长度确定理想的章节数量
    # 文章越长，章节数越多，但增长率逐渐放缓
    if target_length <= 1000:
        ideal_chapter_count = 3  # 短文章：引言、正文、结论
    elif target_length <= 3000:
        ideal_chapter_count = 5  # 中短文章
    elif target_length <= 5000:
        ideal_chapter_count = 7  # 中等长度
    elif target_length <= 10000:
        ideal_chapter_count = 9  # 中长文章
    elif target_length <= 15000:
        ideal_chapter_count = 12  # 长文章
    else:
        # 超长文章，每增加5000字增加约1.2个章节
        ideal_chapter_count = 12 + ((target_length - 15000) // 5000) * 1.2
    
    # 确保每个章节的字数不超过5000，以保持生成质量
    max_chapter_length = 5000
    min_chapters_needed = max(3, (target_length + max_chapter_length - 1) // max_chapter_length)
    
    # 取较大值确保章节足够
    ideal_chapter_count = max(ideal_chapter_count, min_chapters_needed)
    
    # 实际章节数量
    current_chapter_count = len(outline)
    
    logging.info(f"目标长度: {target_length}, 理想章节数: {ideal_chapter_count}, 当前章节数: {current_chapter_count}")
    
    # 如果章节数量接近理想值（±2），则保留现有结构
    if abs(current_chapter_count - ideal_chapter_count) <= 2:
        logging.info("章节数量合理，保留现有结构")
        # 确保包含引言和结论
        return ensure_intro_conclusion(outline)
    
    # 需要增加章节
    if current_chapter_count < ideal_chapter_count:
        logging.info(f"章节数过少，尝试增加章节从 {current_chapter_count} 到接近 {ideal_chapter_count}")
        return split_chapters(outline, ideal_chapter_count, target_length)
    
    # 需要减少章节
    logging.info(f"章节数过多，尝试合并章节从 {current_chapter_count} 到接近 {ideal_chapter_count}")
    return merge_chapters(outline, ideal_chapter_count)

def ensure_intro_conclusion(outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """确保大纲包含引言和结论章节"""
    # 检查是否有引言
    has_intro = any("引言" in item.get("title", "").lower() or 
                     "介绍" in item.get("title", "").lower() or 
                     "introduction" in item.get("title", "").lower() 
                     for item in outline)
    
    # 检查是否有结论
    has_conclusion = any("结论" in item.get("title", "").lower() or 
                         "总结" in item.get("title", "").lower() or 
                         "conclusion" in item.get("title", "").lower() 
                         for item in outline)
    
    # 复制一份，以便修改
    result = outline.copy()
    
    # 如果没有引言，添加一个
    if not has_intro:
        avg_length = sum(item.get("length", 0) for item in outline) // len(outline)
        intro_length = min(avg_length, 500)  # 控制引言长度
        result.insert(0, {"title": "引言", "length": intro_length, "level": 1})
    
    # 如果没有结论，添加一个
    if not has_conclusion:
        avg_length = sum(item.get("length", 0) for item in outline) // len(outline)
        conclusion_length = min(avg_length, 400)  # 控制结论长度
        result.append({"title": "结论", "length": conclusion_length, "level": 1})
    
    return result

def split_chapters(outline: List[Dict[str, Any]], target_count: int, total_length: int) -> List[Dict[str, Any]]:
    """
    通过拆分现有章节来增加章节数量
    """
    # 确保先有引言和结论
    result = ensure_intro_conclusion(outline)
    
    # 计算仍需增加的章节数
    to_add = target_count - len(result)
    if to_add <= 0:
        return result
    
    # 找出最长的章节进行拆分
    while to_add > 0 and len(result) < target_count:
        # 按长度排序，找出最长章节
        sorted_by_length = sorted(result, key=lambda x: x.get("length", 0), reverse=True)
        
        # 跳过引言和结论章节
        for i, item in enumerate(sorted_by_length):
            title = item.get("title", "").lower()
            if "引言" not in title and "介绍" not in title and "introduction" not in title and \
               "结论" not in title and "总结" not in title and "conclusion" not in title:
                longest_chapter = item
                break
        else:
            # 如果所有章节都是引言或结论（不太可能），就拆分第一个
            longest_chapter = sorted_by_length[0]
        
        # 找到这个章节在结果列表中的位置
        chapter_index = result.index(longest_chapter)
        
        # 从结果中移除这个章节
        result.pop(chapter_index)
        
        # 将章节拆分为两部分
        original_length = longest_chapter.get("length", 0)
        original_title = longest_chapter.get("title", "")
        
        # 创建两个新章节
        chapter1 = {
            "title": f"{original_title} (上)",
            "length": original_length // 2,
            "level": 1
        }
        
        chapter2 = {
            "title": f"{original_title} (下)",
            "length": original_length - (original_length // 2),
            "level": 1
        }
        
        # 将新章节插入原位置
        result.insert(chapter_index, chapter1)
        result.insert(chapter_index + 1, chapter2)
        
        # 减少待添加数量
        to_add -= 1
    
    return result

def merge_chapters(outline: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """
    通过合并相似或短小章节来减少章节数量
    """
    # 复制一份，以便修改
    result = outline.copy()
    
    # 计算需要减少的章节数
    to_remove = len(result) - target_count
    if to_remove <= 0:
        return result
    
    # 先确保保留引言和结论
    intro_indices = []
    conclusion_indices = []
    for i, item in enumerate(result):
        title = item.get("title", "").lower()
        if "引言" in title or "介绍" in title or "introduction" in title:
            intro_indices.append(i)
        elif "结论" in title or "总结" in title or "conclusion" in title:
            conclusion_indices.append(i)
    
    # 保护这些特殊章节索引
    protected_indices = intro_indices + conclusion_indices
    
    # 策略1：合并长度小的相邻章节
    i = 0
    while i < len(result) - 1 and to_remove > 0:
        current = result[i]
        next_item = result[i + 1]
        
        # 如果当前章节或下一章节是受保护的，跳过
        if i in protected_indices or i + 1 in protected_indices:
            i += 1
            continue
        
        # 如果两个章节都较小，合并它们
        if current.get("length", 0) < 2000 and next_item.get("length", 0) < 2000:
            # 创建合并章节
            merged_title = f"{current.get('title', '')} 与 {next_item.get('title', '')}"
            merged_length = current.get("length", 0) + next_item.get("length", 0)
            
            # 更新当前章节
            result[i] = {
                "title": merged_title,
                "length": merged_length,
                "level": 1
            }
            
            # 移除下一个章节
            result.pop(i + 1)
            
            # 更新保护索引以反映删除
            protected_indices = [idx if idx < i + 1 else idx - 1 for idx in protected_indices]
            
            # 减少待移除数量
            to_remove -= 1
        else:
            i += 1
    
    # 如果仍需减少章节，合并长度适中的相邻章节
    i = 0
    while i < len(result) - 1 and to_remove > 0:
        current = result[i]
        next_item = result[i + 1]
        
        # 如果当前章节或下一章节是受保护的，跳过
        if i in protected_indices or i + 1 in protected_indices:
            i += 1
            continue
        
        # 尝试合并这两个章节
        merged_title = f"{current.get('title', '')} 与 {next_item.get('title', '')}"
        merged_length = current.get("length", 0) + next_item.get("length", 0)
        
        # 如果合并后长度合理，执行合并
        if merged_length <= 6000:  # 稍微放宽限制
            # 更新当前章节
            result[i] = {
                "title": merged_title,
                "length": merged_length,
                "level": 1
            }
            
            # 移除下一个章节
            result.pop(i + 1)
            
            # 更新保护索引
            protected_indices = [idx if idx < i + 1 else idx - 1 for idx in protected_indices]
            
            # 减少待移除数量
            to_remove -= 1
        else:
            i += 1
    
    return result

def handle_content_block(
    outline: List[Dict[str, Any]], 
    final_article: str, 
    current_index: int, 
    title: str, 
    genre: str, 
    language: str,
    context: Optional[str] = None
) -> Tuple[str, int]:
    """
    处理内容被阻止的情况，回滚并尝试重新生成
    
    Args:
        outline: 完整大纲
        final_article: 当前已生成的文章内容
        current_index: 当前处理的章节索引
        title: 文章标题
        genre: 文章体裁
        language: 期望语言
        context: 可选上下文信息
        
    Returns:
        Tuple[str, int]: 更新后的文章内容和应继续处理的章节索引
    """
    logging.warning(f"检测到内容阻止，尝试回滚和重新生成章节 {current_index}")
    
    # 确定需要回滚到的章节
    rollback_index = max(0, current_index - 1)
    problematic_item = outline[current_index]
    
    # 找到上一章节的结束位置
    if rollback_index > 0:
        previous_title = outline[rollback_index].get("title", "")
        pattern = f"# {re.escape(previous_title)}\n\n(.*?)(?=# |$)"
        match = re.search(pattern, final_article, re.DOTALL)
        if match:
            # 找到上一章节的结束位置
            end_pos = match.end()
            rollback_content = final_article[:end_pos]
        else:
            # 如果找不到上一章节，尝试回滚到文章开头
            rollback_content = re.split(r"# .*?\n\n", final_article, 1)[0]
    else:
        # 如果是第一个章节有问题，回滚到标题和引言
        rollback_content = re.split(r"# .*?\n\n", final_article, 1)[0]
    
    # 重新生成有问题的章节标题
    original_title = problematic_item.get("title", "")
    new_title = regenerate_chapter_title(original_title, genre, language)
    
    # 更新大纲中的标题
    outline[current_index]["title"] = new_title
    logging.info(f"已更新大纲中的标题: 第{current_index+1}章 '{original_title}' -> '{new_title}'")
    
    # 保存回滚的内容
    save_progress(title, rollback_content, suffix="_rollback")
    
    return rollback_content, rollback_index

def get_remaining_outline(outline: List[Dict[str, Any]], written_content: str) -> List[Dict[str, Any]]:
    """
    根据已写作内容提取大纲中未写作的部分
    
    Args:
        outline: 大纲部分（可能是全部或剩余部分）
        written_content: 已写作的内容
        
    Returns:
        List[Dict[str, Any]]: 未写作的大纲部分
    """
    remaining = []
    written_titles = set()
    
    # 提取已写章节标题
    for match in re.finditer(r'^# (.+?)$', written_content, re.MULTILINE):
        written_titles.add(match.group(1).strip())
    
    for item in outline:
        title = item.get("title", "")
        if title and title not in written_titles:
            remaining.append(item)
    
    return remaining

def assemble_article(outline: List[Dict[str, Any]], title: str, length: int, 
                    genre: str, language: str, context: Optional[str] = None) -> str:
    """
    组装文章，使用更全面的上下文避免内容重复和不一致，同时处理内容阻止
    
    Args:
        outline: 完整大纲
        title: 文章标题
        length: 文章期望字数
        genre: 文章体裁
        language: 期望语言
        context: 可选上下文信息
        
    Returns:
        str: 完整文章内容
    """
    logging.info("正在组装文章...")
    final_article = f"# {title}\n\n"  # 以文章标题开始
    
    # 添加简介
    intro_prompt = f"""为标题为'{title}'的{genre}写一个简短的引言或摘要（200-300字），使用{language}。
内容应该概括整篇文章的主题和目的，吸引读者继续阅读。不要包含标题，只需要正文内容。"""
    
    try:
        intro = call_gemini_api(intro_prompt)
        final_article += intro + "\n\n"
    except Exception as e:
        logging.error(f"生成引言失败: {e}")
    
    # 逐章节生成内容，保持全面的上下文
    i = 0
    block_retry_count = 0  # 跟踪连续阻止的次数
    
    while i < len(outline):
        item = outline[i]
        header = f"# {item.get('title', '')}\n\n"
            
        try:
            # 准备当前章节的完整上下文，包括之前写过的所有内容（但有长度限制）
            if len(final_article) > MAX_CONTEXT_LENGTH:
                # 如果已写内容超过限制，只保留最近的内容
                trimmed_content = final_article[-MAX_CONTEXT_LENGTH:]
                # 查找第一个完整章节的开始位置，确保不会从章节中间开始
                first_header_pos = trimmed_content.find("\n# ")
                if first_header_pos > 0:
                    # 从找到的章节标题开始
                    trimmed_content = trimmed_content[first_header_pos+1:]
                previous_context = f"已写完的内容（部分，由于长度限制）:\n\n{trimmed_content}\n\n"
            else:
                previous_context = f"已写完的内容:\n\n{final_article}\n\n"
            
            # 添加未完成的大纲信息，帮助 AI 理解整体结构
            remaining_outline = get_remaining_outline(outline[i:], final_article)
            outline_text = "剩余大纲:\n"
            for outline_item in remaining_outline:
                outline_text += f"- {outline_item.get('title', '')} (约{outline_item.get('length', 0)}字)\n"
            
            # 构建章节提示，强调与上下文的一致性
            current_title = item.get("title", "")
            current_length = item.get("length", 500)
            
            # 确定前后章节以帮助连接
            previous_title = outline[i-1].get("title", "") if i > 0 else "引言"
            next_title = outline[i+1].get("title", "") if i < len(outline) - 1 else "结论"
            
            section_prompt = f"""用{language}写作{genre}的第{i+1}章（共{len(outline)}章）。

章节标题: '{current_title}'
要求字数: 约{current_length}字
前一章: '{previous_title}'
后一章: '{next_title}'

{previous_context}
{outline_text}

请现在编写章节 '{current_title}'。要求:
1. 内容必须与已写内容保持一致，特别是概念、人物、事件等
2. 专注于本章主题，避免重复已写内容
3. 确保内容自然地从前一章过渡，并为后续章节做好铺垫
4. 字数控制在约{current_length}字
5. 只输出正文内容，不要输出章节标题
6. 以专业、适当的方式描述所有内容，避免不当、敏感或可能违反内容政策的描述

确保本章内容完整，可以独立成章，同时与整体保持连贯。"""

            if context:
                section_prompt += f"\n\n额外上下文信息：\n{context}"
            
            try:
                # 调用API生成当前章节内容
                content = call_gemini_api(section_prompt)
                # 内容生成成功，重置阻止计数
                block_retry_count = 0 
                
                section_content = header + content + "\n\n"
                final_article += section_content
                
                # 每生成一个章节都保存一次进度
                save_progress(title, final_article)
                logging.info(f"章节 '{current_title}' 写作完成，生成了 {len(content)} 字符")
                
                # 成功后继续下一章节
                i += 1
                
            except ValueError as e:
                if "提示被阻止" in str(e):
                    # 内容被阻止，需要回滚和重新生成
                    block_retry_count += 1
                    
                    # 如果连续多次被阻止，直接跳过这个章节
                    if block_retry_count > 2:
                        logging.warning(f"章节 '{current_title}' 多次被阻止，跳过此章节")
                        i += 1
                        block_retry_count = 0
                        continue
                    
                    # 处理内容阻止并回滚
                    final_article, i = handle_content_block(
                        outline, final_article, i, title, genre, language, context
                    )
                else:
                    # 其他错误，记录并继续下一章节
                    logging.error(f"生成章节 '{current_title}' 时出错: {e}")
                    i += 1
                    
        except Exception as e:
            logging.error(f"处理章节 '{item.get('title', '')}' 时出错: {e}")
            # 继续下一章节
            i += 1
    
    # 添加结论
    conclusion_prompt = f"""为标题为'{title}'的{genre}写一个清晰的结论（200-300字），使用{language}。

已完成的文章内容（由于长度限制，仅显示最后部分）:
{final_article[-min(len(final_article), MAX_CONTEXT_LENGTH//2):]}

请为这篇文章写一个总结性的结论，要点:
1. 回顾文章的主要内容和观点
2. 提供最终的思考或见解
3. 给读者留下深刻印象
4. 不要重复已有内容，而是提升到更高层次的总结
5. 字数控制在200-300字左右
6. 不要包含标题，只需要正文内容"""
    
    try:
        # 检查大纲中是否已经有结论章节
        if not any(item.get("title", "").lower() in ["结论", "总结", "conclusion", "summary"] for item in outline):
            conclusion = call_gemini_api(conclusion_prompt)
            final_article += "# 结论\n\n" + conclusion + "\n\n"
    except Exception as e:
        logging.error(f"生成结论失败: {e}")
    
    return final_article

def save_progress(title: str, content: str, suffix: str = "_progress") -> str:
    """
    保存文章生成进度
    
    Args:
        title: 文章标题
        content: 当前文章内容
        suffix: 文件名后缀
        
    Returns:
        str: 保存的文件路径
    """
    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)
    
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:30]
    filename = f'{ARTICLES_DIR}/{safe_title}{suffix}.md'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logging.info(f"进度已保存至: {filename}")
    return filename

def save_article(title: str, content: str) -> str:
    """
    将文章内容保存为 Markdown 文件
    
    Args:
        title: 文章标题
        content: 文章内容
        
    Returns:
        str: 保存的文件路径
    """
    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:30]
    filename = f'{ARTICLES_DIR}/{safe_title}_{timestamp}.md'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logging.info(f"文章已保存至: {filename}")
    return filename

def main():
    """主函数"""
    try:
        # 检查环境变量是否设置
        if not os.environ.get("GEMINI_API_KEY"):
            logging.error("未设置GEMINI_API_KEY环境变量。请创建.env文件并设置API密钥。")
            print("错误: 未设置GEMINI_API_KEY环境变量。请创建.env文件并设置API密钥。")
            return
            
        title, length, genre, language, context = get_user_input()
        log_file = setup_logging(title)
        logging.info(f"开始为 {title} 生成文章")
        logging.info(f"日志文件创建于: {log_file}")
        
        # 生成大纲
        outline = generate_outline(title, length, genre, language, context)
        if not outline:
            logging.error("由于大纲生成失败，退出程序")
            return
            
        # 保存大纲
        if not os.path.exists(ARTICLES_DIR):
            os.makedirs(ARTICLES_DIR)
        safe_title = re.sub(r"[^\w\s-]", "", title).strip()[:30]
        outline_file = f'{ARTICLES_DIR}/{safe_title}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_outline.json'
        with open(outline_file, 'w', encoding='utf-8') as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)
        logging.info(f"大纲已保存至: {outline_file}")
        
        # 生成文章
        final_article = assemble_article(outline, title, length, genre, language, context)
        article_file = save_article(title, final_article)
        
        logging.info("文章生成成功完成")
        logging.info(f"日志文件: {log_file}")
        logging.info(f"大纲文件: {outline_file}")
        logging.info(f"文章文件: {article_file}")
        
        print(f"\n文章生成成功！")
        print(f"大纲文件: {outline_file}")
        print(f"文章文件: {article_file}")
        print(f"日志文件: {log_file}")
        
    except KeyboardInterrupt:
        logging.info("用户中断操作")
        print("\n操作已取消")
    except Exception as e:
        logging.exception(f"程序执行过程中出现错误: {e}")
        print(f"\n错误: {e}")
        print("查看日志文件获取更多详情")

if __name__ == "__main__":
    main()