"""Open the project workbench after Slicer's module registry is ready.

This script is intentionally driven by QTimer callbacks.  Slicer discovers
additional modules asynchronously, so a fixed sleep is not reliable across
machines or cold starts.
"""

from __future__ import annotations

import traceback
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
    details = (
        "模块名称：{0}\n"
        "模块路径：{1}\n\n"
        "原因：{2}\n\n"
        "已注册模块（部分）：{3}\n\n"
        "详细异常：\n{4}"
    ).format(
        MODULE_NAME,
        MODULE_PATH,
        reason,
        ", ".join(_module_names()[-30:]),
        _last_error or "未提供额外异常。",
    )
    parent = slicer.util.mainWindow()
    box = qt.QMessageBox(parent)
    box.setIcon(qt.QMessageBox.Critical)
    box.setWindowTitle("下颌髁突标注工具加载失败")
    box.setText("下颌髁突标注工具加载失败。")
    box.setInformativeText("请点击“查看详细信息”获取诊断信息。")
    detail_button = box.addButton("查看详细信息", qt.QMessageBox.AcceptRole)
    close_button = box.addButton("关闭", qt.QMessageBox.RejectRole)
    box.setDefaultButton(close_button)
    if hasattr(box, "exec"):
        box.exec()
    else:
        box.exec_()
    if box.clickedButton() != detail_button:
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
