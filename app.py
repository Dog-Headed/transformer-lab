"""Project entry point: python app.py starts the training API and website."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="启动 Transformer Training Lab 后端与训练网页")
    parser.add_argument("--no-browser", action="store_true", help="启动服务但不自动打开浏览器")
    parser.add_argument("--stop", action="store_true", help="停止本项目启动的服务")
    args = parser.parse_args(argv)

    if os.name != "nt":
        parser.exit(1, "此一键入口用于 Windows；其他系统请按 README 启动 Python 后端和网页。\n")

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = system_root / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not powershell.is_file():
        detected = shutil.which("powershell.exe")
        if not detected:
            parser.exit(1, "启动失败：未找到 Windows PowerShell。\n")
        powershell = Path(detected)

    # Reuse the validated launcher: project-local Python, port ownership,
    # readiness checks, background processes, and logs have one implementation.
    script = PROJECT_ROOT / "scripts" / ("stop.ps1" if args.stop else "start.ps1")
    if not script.is_file():
        parser.exit(1, f"启动文件缺失：{script}\n")
    command = [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if args.no_browser and not args.stop:
        command.append("-NoBrowser")
    print("正在停止服务……" if args.stop else "正在启动服务，首次启动可能需要几分钟……", flush=True)
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, KeyboardInterrupt) as exc:
        parser.exit(1, f"启动被中断或失败：{exc}\n可运行 python app.py --stop 停止已启动的服务。\n")
    if result.returncode:
        parser.exit(result.returncode, f"操作失败，请查看上述提示或 {PROJECT_ROOT / '.runtime'} 中的日志。\n")
    if not args.stop:
        print("启动完成：http://localhost:5173/training", flush=True)
        print("服务在后台运行；停止时运行 python app.py --stop。", flush=True)
    return 0


if __name__ == "__main__":
    # Windows pipes (including IDE consoles) may otherwise default to GBK.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
