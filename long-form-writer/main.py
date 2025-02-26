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
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a log filename based on the article title and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:30]  # Sanitize title for filename
    log_filename = f'logs/{safe_title}_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
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
    """Parses a nested outline string into a list of dictionaries."""
    logging.info(f"Outline text to parse:\n{outline_text}")
    outline = []
    lines = outline_text.strip().split('\n')
    stack = []  # Use a stack to keep track of parent items

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\s*)(\d+(\.\d+)*)\.\s*(.*?)\s*\(约\s*(\d+)\s*字\)$", line)
        if not match:
            continue # Skip lines that don't match the format

        indent = len(match.group(1))
        numbering = match.group(2)
        title = match.group(4).strip()
        word_count = int(match.group(5))
        level = len(numbering.split("."))

        item = {"title": title, "length": word_count, "content": "", "level": level, "children": []}

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if stack:
            stack[-1]["children"].append(item)
        else:
            outline.append(item)
        stack.append(item)

    logging.info(f"Parsed outline: {json.dumps(outline, ensure_ascii=False, indent=2)}")
    return outline

def generate_outline(title, length, genre, language, context):
    """Generates a detailed outline for the article using the Gemini API, requesting JSON output."""
    logging.info(f"Generating outline for title: {title}, length: {length}, genre: {genre}, language: {language}")
    prompt = f"""Create a detailed outline for a {genre} titled '{title}' with a total length of approximately {length} words.

The outline should include:

*   Sections and sub-sections (at least two levels deep).
*   Titles for each section and sub-section.
*   Approximate word counts for each section and sub-section.
*   Ensure the word counts are balanced across sections and sub-sections, reflecting the relative importance and complexity of each topic. The total word count should add up to approximately {length} words.

Output the outline in JSON format as a list of objects. Each object should have the following keys:

*   "title": (string) The title of the section/sub-section.
*   "length": (integer) The approximate word count for the section/sub-section.
*   "level": (integer) The level of the section (1 for top-level, 2 for sub-section, etc.).
*   "parent": (string, optional) The title of the parent section.  Only include for sub-sections.

Example:

[
    {{"title": "Introduction", "length": 100, "level": 1}},
    {{"title": "Background", "length": 200, "level": 1}},
    {{"title": "Historical Context", "length": 150, "level": 2, "parent": "Background"}},
    {{"title": "Modern Developments", "length": 50, "level": 2, "parent": "Background"}},
    {{"title": "Conclusion", "length": 100, "level": 1}}
]
"""
    if context:
        prompt += f"\n\nAdditional context for outline generation:\n{context}"

    try:
        outline_text = call_gemini_api(prompt)
        logging.info(f"Outline text:\n{outline_text}")

        # Extract JSON string from the response
        match = re.search(r"```json\n(.*?)```", outline_text, re.DOTALL)
        if match:
            json_string = match.group(1)
            outline = json.loads(json_string)
            logging.info(f"Parsed outline: {json.dumps(outline, ensure_ascii=False, indent=2)}")
            logging.info("Outline generated successfully.")
            return outline
        else:
            logging.error("Error: Could not find JSON data in API response.")
            return []

    except Exception as e:
        logging.error(f"Error generating outline: {e}")
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
            if "429" in str(e) and attempt < max_retries - 1:  # Check for rate limit error
                wait_time = 2 ** attempt + random.uniform(0, 2)  # Exponential backoff with jitter
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise other exceptions

    raise Exception("Failed to call Gemini API after multiple retries.")

def write_section(title, length, genre, language, outline, written_content, parent_title=None, context=None):
    """Writes a single section of the article with enhanced context awareness."""
    # Only include unwritten sections in the outline context
    remaining_outline = get_remaining_outline(outline, written_content)
    
    # Convert remaining outline to a readable format for the prompt
    outline_text = "文章大纲（待写作部分）：\n"
    for item in remaining_outline:
        indent = "  " * (item["level"] - 1)
        outline_text += f"{indent}- {item['title']} (约{item['length']}字)\n"
    
    # Build the context from previously written content
    previous_content = "已完成的内容：\n\n" if written_content else ""
    if written_content:
        previous_content += written_content

    # Construct the main prompt with clearer section context
    prompt = f"""用{language}写一段{genre}。

当前写作进度：
{previous_content}

剩余大纲：
{outline_text}

请现在写作以下部分（约{length}字）：'{title}'"""

    if parent_title:
        prompt += f"\n这是'{parent_title}'的子部分。"

    prompt += """
要求：
1. 严格保持与已完成内容的连贯性
2. 仅写作指定的当前部分，不要提前写作其他部分
3. 确保内容与大纲规划的主题一致
4. 不要输出任何章节标题，只输出正文内容
"""

    if context:
        prompt += f"\n额外上下文信息：\n{context}"

    logging.info(f"Writing section: {title}")
    logging.debug(f"Full prompt for section {title}:\n{prompt}")
    
    content = call_gemini_api(prompt)
    return content

def get_remaining_outline(outline, written_content):
    """Extract unwritten sections from the outline based on written content."""
    remaining = []
    
    # Create a set of written section titles by parsing the written content
    written_titles = set()
    title_pattern = re.compile(r'^#{1,3}\s+(.+?)\s*$', re.MULTILINE)
    for match in title_pattern.finditer(written_content):
        written_titles.add(match.group(1).strip())
    
    # Filter out sections that have already been written
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
    current_level_2_title = ""

    for item in outline:
        # Skip if content already exists in written_content
        if written_content and f"# {item['title']}" in written_content:
            continue
            
        # Add section header
        if item["level"] == 1:
            header = f"# {item['title']}\n\n"
            current_level_1_title = item["title"]
        elif item["level"] == 2:
            header = f"## {item['title']}\n\n"
            current_level_2_title = item["title"]
        elif item["level"] == 3:
            header = f"### {item['title']}\n\n"
        
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
        
        # Update written content and final article
        written_content += header + content + "\n\n"
        final_article += header + content + "\n\n"

    return final_article

def save_article(title, content):
    """Saves the article content to a markdown file."""
    # Create output directory if it doesn't exist
    if not os.path.exists('articles'):
        os.makedirs('articles')
    
    # Create filename based on title and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:30]  # Sanitize title for filename
    filename = f'articles/{safe_title}_{timestamp}.md'
    
    # Save the content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logging.info(f"Article saved to: {filename}")
    return filename

def main():
    """Main function."""
    title, length, genre, language, context = get_user_input()
    
    # Setup logging
    log_file = setup_logging(title)
    logging.info(f"Starting article generation for: {title}")
    logging.info(f"Log file created at: {log_file}")
    
    # Generate outline
    outline = generate_outline(title, length, genre, language, context)
    if not outline:
        logging.error("Exiting due to outline generation failure.")
        return

    # Save outline to JSON file
    if not os.path.exists('articles'):
        os.makedirs('articles')

    # 清理标题并生成文件名
    safe_title = re.sub(r"[^\w\s-]", "", title).strip()[:30]  # 清理标题
    outline_file = f'articles/{safe_title}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_outline.json'
    with open(outline_file, 'w', encoding='utf-8') as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)
    logging.info(f"Outline saved to: {outline_file}")

    # Generate article with enhanced context awareness
    final_article = assemble_article(outline, title, length, genre, language, context)
    
    # Save the final article
    article_file = save_article(title, final_article)
    
    logging.info("Article generation completed successfully")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Outline file: {outline_file}")
    logging.info(f"Article file: {article_file}")

if __name__ == "__main__":
    main()