#!/usr/bin/env python3
"""
运行脚本：启动Flask后端服务和Web前端

此脚本会启动文章生成器的后端服务和前端界面，便于用户使用。
"""

import os
import sys
import subprocess
import webbrowser
import time
import signal
import http.server
import socketserver
import threading
from pathlib import Path

# 配置
FLASK_SERVER_PORT = 5000
WEB_SERVER_PORT = 80
FLASK_SCRIPT = "flask_server.py"  # Flask后端脚本
HTML_FILE = "article_generator.html"  # 前端HTML文件

def check_requirements():
    """检查所需依赖是否已安装"""
    try:
        import flask
        import flask_cors
        return True
    except ImportError:
        print("缺少必要的依赖，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
        return True

def start_flask_server():
    """启动Flask后端服务器"""
    print("正在启动后端服务器...")
    
    # 检查Flask脚本是否存在
    if not os.path.exists(FLASK_SCRIPT):
        print(f"错误：找不到Flask服务器脚本 {FLASK_SCRIPT}")
        print("请确保该文件与此脚本位于同一目录下")
        return None
    
    # 启动Flask服务器
    flask_process = subprocess.Popen(
        [sys.executable, FLASK_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 等待服务器启动
    print("等待Flask服务器启动...")
    time.sleep(2)
    
    # 检查服务器是否成功启动
    if flask_process.poll() is not None:
        # 服务器已终止
        stdout, stderr = flask_process.communicate()
        print("Flask服务器启动失败:")
        print(stderr)
        return None
    
    print(f"Flask后端服务器已在 http://127.0.0.1:{FLASK_SERVER_PORT} 启动")
    return flask_process

def start_web_server():
    """启动Web服务器提供前端页面访问"""
    print("正在启动Web服务器...")
    
    # 检查HTML文件是否存在
    if not os.path.exists(HTML_FILE):
        print(f"错误：找不到前端HTML文件 {HTML_FILE}")
        print("请确保该文件与此脚本位于同一目录下")
        return None
    
    # 使用Python内置的HTTP服务器
    handler = http.server.SimpleHTTPRequestHandler
    
    def run_server():
        with socketserver.TCPServer(("", WEB_SERVER_PORT), handler) as httpd:
            print(f"Web服务器已在 http://127.0.0.1:{WEB_SERVER_PORT} 启动")
            httpd.serve_forever()
    
    # 在新线程中启动服务器
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    return server_thread

def open_browser():
    """打开浏览器访问前端页面"""
    print("正在打开浏览器...")
    url = f"http://127.0.0.1:{WEB_SERVER_PORT}/{HTML_FILE}"
    webbrowser.open(url)

def main():
    """主函数"""
    print("===== 文章生成器启动脚本 =====")
    
    # 检查依赖
    if not check_requirements():
        return
    
    # 启动Flask后端
    flask_process = start_flask_server()
    if not flask_process:
        return
    
    # 启动Web服务器
    server_thread = start_web_server()
    if not server_thread:
        flask_process.terminate()
        return
    
    # 等待服务器完全启动
    time.sleep(1)
    
    # 打开浏览器
    open_browser()
    
    print("\n服务已启动！请在浏览器中使用文章生成器")
    print("按 Ctrl+C 停止服务\n")
    
    try:
        # 保持程序运行，直到用户按Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n接收到终止信号，正在关闭服务...")
        # 终止Flask进程
        if flask_process:
            flask_process.terminate()
            flask_process.wait()
        print("服务已停止")

if __name__ == "__main__":
    main()