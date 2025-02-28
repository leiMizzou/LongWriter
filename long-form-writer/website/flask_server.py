import os
import sys
import json
import time
import tempfile
import subprocess
import threading
import re
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求，方便本地开发

# 文章生成器脚本路径
ARTICLE_GENERATOR_PATH = "article_generator.py"  # 替换为您的实际脚本名称

# 临时目录，用于存储上传的模板文件
TEMP_DIR = tempfile.gettempdir()

# 全局任务状态存储
class TaskStatus:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
    
    def create_task(self, task_id):
        with self.lock:
            self.tasks[task_id] = {
                "status": "initializing",
                "progress": 0,
                "message": "正在初始化...",
                "logs": [],
                "outline": None,
                "article_file": None,
                "outline_file": None,
                "log_file": None,
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None
            }
            return self.tasks[task_id]
    
    def update_task(self, task_id, **kwargs):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                return True
            return False
    
    def add_log(self, task_id, log_line):
        with self.lock:
            if task_id in self.tasks:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.tasks[task_id]["logs"].append(f"[{timestamp}] {log_line}")
                
                # 根据日志内容更新进度状态
                if "正在生成大纲" in log_line:
                    self.tasks[task_id]["status"] = "generating_outline"
                    self.tasks[task_id]["progress"] = 10
                    self.tasks[task_id]["message"] = "正在生成大纲..."
                elif "大纲生成成功" in log_line:
                    self.tasks[task_id]["status"] = "outline_completed"
                    self.tasks[task_id]["progress"] = 20
                    self.tasks[task_id]["message"] = "大纲生成完成，开始撰写文章..."
                elif "正在组装文章" in log_line:
                    self.tasks[task_id]["status"] = "generating_article"
                    self.tasks[task_id]["progress"] = 25
                    self.tasks[task_id]["message"] = "开始生成文章内容..."
                elif "章节" in log_line and "写作完成" in log_line:
                    # 提取章节信息，更新进度
                    chapter_match = re.search(r"章节\s+'(.+?)'\s+写作完成", log_line)
                    if chapter_match:
                        chapter = chapter_match.group(1)
                        # 更新进度，根据已完成章节估算（25%是起点，90%是终点）
                        current_progress = self.tasks[task_id]["progress"]
                        if current_progress < 90:
                            self.tasks[task_id]["progress"] = min(current_progress + 5, 90)
                        self.tasks[task_id]["message"] = f"正在写作章节: {chapter}"
                elif "文章生成成功" in log_line:
                    self.tasks[task_id]["status"] = "completed"
                    self.tasks[task_id]["progress"] = 100
                    self.tasks[task_id]["message"] = "文章生成成功！"
                    self.tasks[task_id]["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elif "错误" in log_line.lower() or "fail" in log_line.lower() or "error" in log_line.lower():
                    self.tasks[task_id]["status"] = "error"
                    self.tasks[task_id]["message"] = f"发生错误: {log_line}"
                
                # 识别文件路径
                if "文章文件:" in log_line:
                    self.tasks[task_id]["article_file"] = log_line.split("文章文件:")[1].strip()
                elif "大纲文件:" in log_line:
                    self.tasks[task_id]["outline_file"] = log_line.split("大纲文件:")[1].strip()
                    # 尝试读取大纲文件内容
                    outline_file = log_line.split("大纲文件:")[1].strip()
                    if os.path.exists(outline_file):
                        try:
                            with open(outline_file, 'r', encoding='utf-8') as f:
                                self.tasks[task_id]["outline"] = json.load(f)
                        except Exception as e:
                            print(f"读取大纲文件失败: {e}")
                elif "日志文件:" in log_line:
                    self.tasks[task_id]["log_file"] = log_line.split("日志文件:")[1].strip()
                
                return True
            return False
    
    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)
    
    def cleanup_old_tasks(self, hours=24):
        """清理超过指定小时数的旧任务"""
        with self.lock:
            now = datetime.now()
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.get("finished_at"):
                    finished_time = datetime.strptime(task["finished_at"], "%Y-%m-%d %H:%M:%S")
                    if (now - finished_time).total_seconds() > hours * 3600:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]

# 创建任务状态管理器实例
task_manager = TaskStatus()

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取服务器状态"""
    return jsonify({
        'status': 'ready',
        'message': '服务就绪'
    })

@app.route('/api/generate', methods=['POST'])
def generate_article():
    """处理文章生成请求并在后台执行"""
    # 生成唯一任务ID
    task_id = f"task_{int(time.time())}_{os.urandom(4).hex()}"
    
    # 初始化任务状态
    task = task_manager.create_task(task_id)
    
    # 创建后台线程执行生成任务
    thread = threading.Thread(
        target=run_generation_task,
        args=(task_id, request.form.to_dict(), dict(request.files))
    )
    thread.daemon = True
    thread.start()
    
    # 返回任务ID供客户端查询进度
    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': '任务已提交，正在处理中'
    })

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务的当前状态和进度"""
    task = task_manager.get_task(task_id)
    if task:
        return jsonify({
            'success': True,
            'task': task
        })
    else:
        return jsonify({
            'success': False,
            'error': '找不到指定的任务'
        }), 404

@app.route('/api/download/<file_type>', methods=['GET'])
def download_file(file_type):
    """处理文件下载请求"""
    file_path = request.args.get('path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 404
    
    return send_file(file_path, as_attachment=True)

def handle_output(pipe, task_id, task_manager, is_error=False):
    """处理进程输出流，区分正常日志和错误信息"""
    for line in iter(pipe.readline, ''):
        line = line.strip()
        if line:
            # 不再自动将stderr标记为错误，而是分析日志内容
            if (is_error and ('error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower())):
                # 这是真正的错误信息
                task_manager.add_log(task_id, f"错误: {line}")
            else:
                # 这是普通日志
                task_manager.add_log(task_id, line)

def run_generation_task(task_id, form_data, files):
    """在后台线程中执行文章生成任务，改进错误处理"""
    try:
        # 保存完整的过程输出到文件(用于调试)
        debug_log_file = f"debug_process_{task_id}.log"
        
        # 更新任务状态
        task_manager.update_task(task_id, status="preparing", progress=5, message="正在准备生成环境...")
        
        # 获取表单数据
        title = form_data.get('title')
        length = form_data.get('length')
        genre = form_data.get('genre')
        language = form_data.get('language')
        context = form_data.get('context', '')
        
        # 检查脚本路径是否存在
        if not os.path.exists(ARTICLE_GENERATOR_PATH):
            error_msg = f"错误: 找不到文章生成脚本: {ARTICLE_GENERATOR_PATH}"
            task_manager.add_log(task_id, error_msg)
            task_manager.update_task(
                task_id,
                status="error",
                progress=100,
                message="脚本文件不存在"
            )
            return
        
        # 设置命令行参数
        python_executable = sys.executable
        task_manager.add_log(task_id, f"使用Python解释器: {python_executable}")
        
        cmd = [python_executable, ARTICLE_GENERATOR_PATH, title, length, genre, language]
        
        # 记录完整命令
        task_manager.add_log(task_id, f"执行命令: {' '.join(cmd)}")
        
        # 检查必要目录是否存在
        articles_dir = os.path.join(os.path.dirname(os.path.abspath(ARTICLE_GENERATOR_PATH)), "articles")
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(ARTICLE_GENERATOR_PATH)), "logs")
        
        if not os.path.exists(articles_dir):
            task_manager.add_log(task_id, f"创建文章目录: {articles_dir}")
            os.makedirs(articles_dir)
            
        if not os.path.exists(logs_dir):
            task_manager.add_log(task_id, f"创建日志目录: {logs_dir}")
            os.makedirs(logs_dir)
        
        # 处理可选参数
        if context:
            cmd.extend(["--context", context])
            task_manager.add_log(task_id, f"添加上下文参数，长度: {len(context)} 字符")
        
        # 处理模板文件
        template_path = None
        if 'template_file' in files:
            template_file = files['template_file'][0]
            if template_file.filename:
                template_path = os.path.join(TEMP_DIR, secure_filename(template_file.filename))
                template_file.save(template_path)
                cmd.extend(["--template", template_path])
                task_manager.add_log(task_id, f"使用模板文件: {template_file.filename}")
        
        # 处理API密钥
        env = os.environ.copy()
        if form_data.get('api_key'):
            env['GEMINI_API_KEY'] = form_data.get('api_key')
            task_manager.add_log(task_id, "使用自定义API密钥")
        elif 'GEMINI_API_KEY' not in env:
            task_manager.add_log(task_id, "警告: 未设置GEMINI_API_KEY环境变量")
        
        # 更新任务状态
        task_manager.update_task(task_id, status="running", progress=10, message="开始执行文章生成...")
        
        # 获取当前脚本的绝对路径目录
        script_dir = os.path.dirname(os.path.abspath(ARTICLE_GENERATOR_PATH))
        task_manager.add_log(task_id, f"工作目录: {script_dir}")
        
        # 创建进程执行脚本，并实时捕获输出
        with open(debug_log_file, "w", encoding="utf-8") as debug_file:
            debug_file.write(f"==== DEBUG LOG FOR TASK {task_id} ====\n")
            debug_file.write(f"Command: {' '.join(cmd)}\n")
            debug_file.write(f"Working directory: {script_dir}\n")
            debug_file.write(f"Environment GEMINI_API_KEY set: {'Yes' if 'GEMINI_API_KEY' in env else 'No'}\n")
            debug_file.write("==== PROCESS OUTPUT ====\n")
            debug_file.flush()
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 行缓冲，确保及时获取输出
                cwd=script_dir  # 设置工作目录，确保相对路径正常工作
            )
            
            # 实时处理标准输出和错误，同时记录到调试文件
            def handle_pipe_with_debug(pipe, is_error=False):
                prefix = "STDERR" if is_error else "STDOUT"
                for line in iter(pipe.readline, ''):
                    if line.strip():
                        # 写入调试文件
                        debug_file.write(f"[{prefix}] {line}")
                        debug_file.flush()
                        
                        # 处理用于状态更新的输出
                        if (is_error and ('error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower())):
                            task_manager.add_log(task_id, f"错误: {line.strip()}")
                        else:
                            task_manager.add_log(task_id, line.strip())
            
            # 创建线程处理标准输出和标准错误
            stdout_thread = threading.Thread(
                target=handle_pipe_with_debug, 
                args=(process.stdout,)
            )
            stderr_thread = threading.Thread(
                target=handle_pipe_with_debug, 
                args=(process.stderr, True)
            )
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # 等待进程完成
            return_code = process.wait()
            stdout_thread.join()
            stderr_thread.join()
            
            debug_file.write(f"==== PROCESS FINISHED WITH CODE {return_code} ====\n")
            
            # 检查是否成功
            if return_code != 0:
                task_manager.update_task(
                    task_id,
                    status="error",
                    progress=100,
                    message=f"执行失败，返回代码: {return_code}"
                )
                task_manager.add_log(task_id, f"进程异常退出，返回代码: {return_code}")
                
                # 记录额外的错误信息
                debug_file.write("==== ATTEMPT TO CAPTURE FINAL ERROR OUTPUT ====\n")
                try:
                    _, stderr_output = process.communicate(timeout=1)
                    if stderr_output:
                        debug_file.write(stderr_output)
                        task_manager.add_log(task_id, f"错误输出: {stderr_output}")
                except:
                    debug_file.write("Failed to capture final error output\n")
                    pass
            else:
                # 验证是否成功生成了文件
                article_file = None
                outline_file = None
                log_file = None
                
                # 查找任务状态中已记录的文件路径
                task = task_manager.get_task(task_id)
                article_file = task.get('article_file')
                outline_file = task.get('outline_file')
                log_file = task.get('log_file')
                
                debug_file.write("==== CHECKING GENERATED FILES ====\n")
                debug_file.write(f"Article file: {article_file}\n")
                debug_file.write(f"Outline file: {outline_file}\n")
                debug_file.write(f"Log file: {log_file}\n")
                
                # 检查文件是否真的存在
                if article_file and not os.path.exists(article_file):
                    task_manager.add_log(task_id, f"警告: 文章文件不存在: {article_file}")
                    debug_file.write(f"WARNING: Article file does not exist: {article_file}\n")
                    article_file = None
                
                if outline_file and not os.path.exists(outline_file):
                    task_manager.add_log(task_id, f"警告: 大纲文件不存在: {outline_file}")
                    debug_file.write(f"WARNING: Outline file does not exist: {outline_file}\n")
                    outline_file = None
                
                if log_file and not os.path.exists(log_file):
                    task_manager.add_log(task_id, f"警告: 日志文件不存在: {log_file}")
                    debug_file.write(f"WARNING: Log file does not exist: {log_file}\n")
                    log_file = None
                
                # 检查是否生成了任何文件
                if not article_file and not outline_file:
                    task_manager.update_task(
                        task_id,
                        status="error",
                        progress=100,
                        message="执行结束但没有生成文件"
                    )
                    task_manager.add_log(task_id, "进程正常退出，但没有生成任何文件")
                    debug_file.write("ERROR: Process completed but no files were generated\n")
                else:
                    # 如果还没被标记为完成，更新为完成状态
                    if task.get("status") != "completed":
                        task_manager.update_task(
                            task_id,
                            status="completed",
                            progress=100,
                            message="文章生成完成！",
                            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        debug_file.write("SUCCESS: Process completed successfully\n")
        
        # 添加调试日志文件路径到任务状态
        task_manager.update_task(task_id, debug_log=debug_log_file)
        task_manager.add_log(task_id, f"完整调试日志: {debug_log_file}")
        
    except Exception as e:
        # 捕获执行过程中的任何异常
        import traceback
        error_trace = traceback.format_exc()
        
        task_manager.update_task(
            task_id,
            status="error",
            progress=100,
            message=f"执行过程中发生错误: {str(e)}"
        )
        task_manager.add_log(task_id, f"异常: {str(e)}")
        task_manager.add_log(task_id, f"错误详情: {error_trace}")
        
        # 记录错误到调试文件
        try:
            with open(debug_log_file, "a", encoding="utf-8") as debug_file:
                debug_file.write(f"==== BACKEND EXCEPTION ====\n")
                debug_file.write(f"Error: {str(e)}\n")
                debug_file.write(f"Traceback: {error_trace}\n")
        except:
            pass

# 启动定期清理任务的线程
def cleanup_thread():
    while True:
        time.sleep(3600)  # 每小时清理一次
        task_manager.cleanup_old_tasks()

cleanup_thread = threading.Thread(target=cleanup_thread)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == '__main__':
    print("启动文章生成器后端服务...")
    print(f"检查脚本路径: {ARTICLE_GENERATOR_PATH}")
    if not os.path.exists(ARTICLE_GENERATOR_PATH):
        print(f"警告: 找不到文章生成脚本: {ARTICLE_GENERATOR_PATH}")
    
    # 确保必要的目录存在
    if not os.path.exists('articles'):
        print("创建文章目录: articles/")
        os.makedirs('articles')
    
    if not os.path.exists('logs'):
        print("创建日志目录: logs/")
        os.makedirs('logs')
    
    # 检查API密钥
    if 'GEMINI_API_KEY' not in os.environ:
        print("警告: 未设置GEMINI_API_KEY环境变量")
        print("您可以在前端界面中手动输入API密钥，或者创建.env文件设置")
    
    print("后端服务已就绪，正在监听端口5000...")
    app.run(debug=True, port=5000, threaded=True)