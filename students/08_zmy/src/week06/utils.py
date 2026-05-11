import shutil
from pathlib import Path

def setup_results_dir() -> Path:
    """
    在代码所在目录（students/08_zmy/src/week06）下创建 results 文件夹。
    """
    code_dir = Path(__file__).resolve().parent   # week06 目录
    results_dir = code_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def save_markdown_table_header(file, title, headers):
    """写入 Markdown 表格头。"""
    file.write(f"# {title}\n\n")
    file.write("| " + " | ".join(headers) + " |\n")
    file.write("|" + "|".join([" --- " for _ in headers]) + "|\n")