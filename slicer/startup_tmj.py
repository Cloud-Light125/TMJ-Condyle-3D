"""Open the project workbench after Slicer's module registry is ready.

This script is intentionally driven by QTimer callbacks.  Slicer discovers
additional modules asynchronously, so a fixed sleep is not reliable across
machines or cold starts.
"""

from __future__ import annotations

import traceback
import os
from pathlib import Path

import qt
import slicer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "TMJCondyleAnnotator"
MODULE_PATH = PROJECT_ROOT / "slicer" / "TMJCondyleAnnotator"
MAX_ATTEMPTS = 240
_attempts = 0
_last_error = ""


def _module_names():
    manager = slicer.app.moduleManager()
    names = manager.modulesNames()
    return tuple(str(name) for name in names)


def _show_failure(reason):
    slicer_log, slicer_log_path = _slicer_log_text()
    details = (
        "模块名称：{0}\n"
        "模块路径：{1}\n\n"
        "原因：{2}\n\n"
        "已注册模块（部分）：{3}\n\n"
        "Slicer 日志文件：{4}\n"
        "Slicer 日志（最近内容）：\n{5}\n\n"
        "详细异常：\n{6}"
    ).format(
        MODULE_NAME,
        MODULE_PATH,
        reason,
        ", ".join(_module_names()[-30:]),
        slicer_log_path or "未找到可读的 Slicer 日志。",
        slicer_log or "未找到可读的 Slicer 日志。",
        _last_error or "未提供额外异常。",
    )
    parent = slicer.util.mainWindow()
    box = qt.QMessageBox(parent)
    box.setIcon(qt.QMessageBox.Critical)
    box.setWindowTitle("下颌髁突三维分割实验平台加载失败")
    box.setText("实验平台启动失败，请点击下面按钮查看原因。")
    box.setInformativeText("模块没有完成自动加载，项目文件和启动参数仍会保留。")
    restart_button = box.addButton("重新启动", qt.QMessageBox.AcceptRole)
    detail_button = box.addButton("查看详细信息", qt.QMessageBox.AcceptRole)
    close_button = box.addButton("关闭", qt.QMessageBox.RejectRole)
    box.setDefaultButton(close_button)
    if hasattr(box, "exec"):
        box.exec()
    else:
        box.exec_()
    clicked = box.clickedButton()
    if clicked == restart_button:
        _restart_slicer()
        return
    if clicked != detail_button:
        return
    detail_dialog = qt.QDialog(parent)
    detail_dialog.setWindowTitle("模块加载详细信息")
    detail_dialog.setMinimumSize(760, 440)
    detail_layout = qt.QVBoxLayout(detail_dialog)
    detail_text = qt.QPlainTextEdit()
    detail_text.setReadOnly(True)
    detail_text.setPlainText(details)
    detail_layout.addWidget(detail_text)
    close_detail = qt.QPushButton("关闭")
    close_detail.clicked.connect(lambda checked=False: detail_dialog.accept())
    detail_layout.addWidget(close_detail)
    if hasattr(detail_dialog, "exec"):
        detail_dialog.exec()
    else:
        detail_dialog.exec_()


def _slicer_log_text():
    """Return a short tail of the most likely Slicer log file."""

    candidates = []
    for attribute in ("logFile", "logFileName", "logName"):
        try:
            value = getattr(slicer.app, attribute)
            value = value() if callable(value) else value
            if value:
                candidates.append(Path(str(value)))
        except Exception:
            pass
    for root in (
        Path.cwd(),
        Path(os.environ.get("LOCALAPPDATA", "")) / "NA-MIC",
        Path.home() / "AppData" / "Local" / "NA-MIC",
        Path(slicer.app.applicationDirPath()).parent,
    ):
        if not root or not root.exists():
            continue
        try:
            candidates.extend(root.glob("**/*.log"))
        except Exception:
            pass
    unique = []
    seen = set()
    for path in candidates:
        try:
            path = path.resolve()
            key = str(path).casefold()
            if key not in seen and path.is_file():
                seen.add(key)
                unique.append(path)
        except Exception:
            pass
    unique.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not unique:
        return "", ""
    path = unique[0]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[-8000:]
    except Exception:
        text = ""
    return text, str(path)


def _restart_slicer():
    try:
        restart = getattr(slicer.app, "restart", None)
        if callable(restart):
            restart()
            return
    except Exception:
        pass
    try:
        slicer.app.quit()
    except Exception:
        pass


def _open_workbench_when_ready():
    global _attempts, _last_error
    _attempts += 1
    try:
        names = _module_names()
        if MODULE_NAME in names:
            main_window = slicer.util.mainWindow()
            if main_window:
                if main_window.isMinimized():
                    main_window.showNormal()
                main_window.show()
                main_window.raise_()
            slicer.util.selectModule(MODULE_NAME)
            widget = slicer.util.getModuleWidget(MODULE_NAME)
            if widget is not None:
                return
        if _attempts >= MAX_ATTEMPTS:
            _show_failure("等待模块注册超时。")
            return
    except Exception:
        _last_error = traceback.format_exc()
        if _attempts >= MAX_ATTEMPTS:
            _show_failure("自动进入模块时发生异常。")
            return
    qt.QTimer.singleShot(100, _open_workbench_when_ready)


qt.QTimer.singleShot(0, _open_workbench_when_ready)
