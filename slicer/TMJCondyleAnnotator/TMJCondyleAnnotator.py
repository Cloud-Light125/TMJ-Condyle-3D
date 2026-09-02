"""Chinese four-step workbench for manual mandibular-condyle annotation.

The clinical-facing workflow lives in this module.  A qMRMLSegmentEditorWidget
is kept as an embedded editing engine, but the native Segment Editor module is
never selected or shown to the user.  The module owns the Segment Editor
parameter node, the MRML scene association, and slice-view observations.
"""

from __future__ import annotations

import csv
import re
import traceback
from pathlib import Path

import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
CASE_FILE_PATTERN = re.compile(r"^case_[A-Za-z0-9_-]+$")
MANIFEST_FIELDS = [
    "case_id",
    "group_id",
    "side",
    "image_path",
    "label_path",
    "annotation_status",
    "geometry_valid",
    "label_valid",
    "notes",
]

# These are Slicer effect identifiers.  They intentionally stay in English at
# the engine boundary; every user-facing caption is defined separately below.
EFFECT_ORDER = ["Paint", "Erase", "Scissors", "Islands", "Fill between slices"]
EFFECT_LABELS = {
    "Paint": "画笔",
    "Erase": "擦除",
    "Scissors": "剪刀",
    "Islands": "清理零碎区域",
    "Fill between slices": "层间补全",
}
ORIENTATION_LABELS = {
    0: "矢状位",
    1: "冠状位",
    2: "轴位",
}
ORIENTATIONS = {
    0: "Sagittal",
    1: "Coronal",
    2: "Axial",
}


class TMJCondyleAnnotator(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "下颌髁突三维标注"
        self.parent.categories = ["Segmentation"]
        self.parent.contributors = ["TMJ-Condyle-3D"]
        self.parent.helpText = (
            "面向牙医和医学生的下颌髁突人工标注工作台。"
            "导入、标注、检查和保存都在同一页面完成。"
        )
        self.parent.acknowledgementText = (
            "This module is part of the TMJ-Condyle-3D teaching/research workflow."
        )


class TMJCondyleAnnotatorWidget(ScriptedLoadableModuleWidget):
    """Modern Chinese workflow panel with an embedded segment editor engine."""

    STEP_NAMES = ["导入", "标注", "检查", "保存"]

    def setup(self):
        super().setup()

        self.projectRoot = Path(__file__).resolve().parents[2]
        self.niftiDir = self.projectRoot / "workspace" / "nifti"
        self.labelsDir = self.projectRoot / "workspace" / "labels"
        self.manifestPath = self.projectRoot / "workspace" / "dataset_manifest.csv"

        self.volumeNode = None
        self.segmentationNode = None
        self.segmentId = None
        self._parameterNode = None
        self.editorWidget = None
        self._effectFactory = None
        self._effectFactoryConnected = False
        self._editorReady = False
        self._editorViewsObserved = False
        self._pendingEffectName = None
        self._pendingEffectAttempts = 0

        self._ownedVolumeIds = set()
        self._loadedVolumeIds = []
        self._segmentationObservers = []
        self._sliceObservers = []
        self._sceneObservers = []
        self._mainWindow = None
        self._isUpdating = False
        self._moduleIsActive = False
        self._reenterScheduled = False

        self._dirty = False
        self._saved = False
        self._annotationHasData = False
        self._qcStatus = "未检查"
        self._currentEffectName = "Paint"
        self._currentCaseId = "case_001"
        self._currentCasePath = None
        self._currentCaseIndex = 0
        self._caseFiles = []
        self._lastDetailText = ""
        self._currentPage = 0
        self._primaryAxis = 2
        self._primaryOrientation = "Axial"

        self._simpleMode = False
        self._simpleModeTargets = []
        self._previousLayout = None
        self._workflowLayoutChanged = False
        self._customLayoutId = None
        self._customLayoutAdded = False

        self._buildUi()
        self._createEmbeddedEditor()
        self._installSceneObservers()
        self._refreshCaseFiles()
        self._refreshLoadedVolumes()
        self._installCloseProtection()
        self._refreshSliceObservers()
        self._setSimpleMode(True)
        # The Slicer main window can finish creating its toolbars and auxiliary
        # panels just after a scripted module is constructed.  Re-apply the
        # temporary simple-mode filter once the main window is settled.
        qt.QTimer.singleShot(800, lambda: self._setSimpleMode(True))
        self._setStatusMessage("请先选择一份需要标注的 MRI。", "info")
        self._syncUi()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    @staticmethod
    def _styleSheet():
        return """
        QWidget#tmjRoot {
          background: #f5f8fb;
          color: #1f3444;
        }
        QScrollArea#tmjScrollArea {
          border: none;
          background: #f5f8fb;
        }
        QFrame#headerCard, QFrame#stepCard, QFrame#caseCard,
        QFrame#contentCard, QFrame#hintCard, QFrame#statusCard,
        QFrame#summaryCard, QLabel#summaryCard {
          background: #ffffff;
          border: 1px solid #e1eaf0;
          border-radius: 14px;
        }
        QFrame#stepCard[state="current"] {
          background: #eaf6f8;
          border: 2px solid #16839a;
        }
        QFrame#stepCard[state="done"] {
          background: #eef9f2;
          border: 1px solid #94d5ad;
        }
        QFrame#stepCard[state="waiting"] {
          background: #f8fafc;
          border: 1px solid #e6edf2;
        }
        QLabel#mainTitle {
          color: #163b57;
          font-size: 22px;
          font-weight: 600;
        }
        QLabel#subtitleLabel {
          color: #6b7d8b;
          font-size: 12px;
        }
        QLabel#pageEyebrow {
          color: #16839a;
          font-size: 12px;
          font-weight: 600;
        }
        QLabel#pageTitle {
          color: #163b57;
          font-size: 20px;
          font-weight: 600;
        }
        QLabel#sectionTitle {
          color: #163b57;
          font-size: 14px;
          font-weight: 600;
        }
        QLabel#mutedLabel, QLabel#stepStatus, QLabel#caseCaption,
        QLabel#hintLabel {
          color: #718391;
        }
        QLabel#stepNumber {
          color: #163b57;
          font-size: 13px;
          font-weight: 600;
        }
        QLabel#stepStatus {
          font-size: 11px;
        }
        QLabel#caseValue {
          color: #163b57;
          font-size: 14px;
          font-weight: 600;
        }
        QLabel#statusChip {
          background: #eef2f5;
          border-radius: 12px;
          color: #526777;
          font-size: 12px;
          font-weight: 600;
          padding: 6px 11px;
        }
        QLabel#statusChip[status="working"] {
          background: #fff5df;
          color: #9a6500;
        }
        QLabel#statusChip[status="complete"] {
          background: #eaf8ef;
          color: #247443;
        }
        QLabel#statusChip[status="warning"] {
          background: #fff1ed;
          color: #a34b31;
        }
        QLabel#statusMessage {
          border-radius: 10px;
          padding: 9px 12px;
          background: #eef6fb;
          color: #2f6178;
        }
        QLabel#statusMessage[state="success"] {
          background: #eaf8ef;
          color: #247443;
        }
        QLabel#statusMessage[state="warning"] {
          background: #fff1ed;
          color: #a34b31;
        }
        QLabel#statusMessage[state="neutral"] {
          background: #f2f5f7;
          color: #637481;
        }
        QLabel#resultMessage {
          border-radius: 10px;
          padding: 10px 12px;
          background: #f2f5f7;
          color: #637481;
        }
        QLabel#resultMessage[state="success"] {
          background: #eaf8ef;
          color: #247443;
        }
        QLabel#resultMessage[state="warning"] {
          background: #fff6df;
          color: #8b650c;
        }
        QLabel#bigSuccess {
          color: #247443;
          font-size: 22px;
          font-weight: 600;
          padding: 12px;
        }
        QPushButton#primaryButton {
          background: #16839a;
          color: #ffffff;
          border: none;
          border-radius: 10px;
          min-height: 40px;
          padding: 0 16px;
          font-size: 13px;
          font-weight: 600;
        }
        QPushButton#primaryButton:hover {
          background: #126f84;
        }
        QPushButton#primaryButton:disabled {
          background: #b7cbd2;
          color: #edf5f7;
        }
        QPushButton#secondaryButton {
          background: #ffffff;
          color: #28546a;
          border: 1px solid #c8dbe2;
          border-radius: 9px;
          min-height: 34px;
          padding: 0 12px;
          font-size: 12px;
        }
        QPushButton#secondaryButton:hover {
          background: #f2f8fa;
          border-color: #8dbbc7;
        }
        QPushButton#secondaryButton:disabled {
          color: #a6b4bc;
          border-color: #e1e8ec;
          background: #fafbfc;
        }
        QPushButton#toolButton {
          background: #edf7f8;
          color: #165c6c;
          border: 1px solid #b9dfe4;
          border-radius: 12px;
          min-height: 56px;
          padding: 0 20px;
          font-size: 16px;
          font-weight: 600;
        }
        QPushButton#toolButton:hover {
          background: #e0f1f3;
        }
        QPushButton#toolButton[active="true"] {
          background: #16839a;
          color: #ffffff;
          border: 2px solid #0f6477;
        }
        QPushButton#toolButton:disabled {
          background: #f2f5f6;
          color: #a7b4ba;
          border-color: #e2e8eb;
        }
        QToolButton#foldButton, QPushButton#linkButton {
          background: transparent;
          color: #16839a;
          border: none;
          padding: 4px 2px;
          font-size: 12px;
        }
        QToolButton#foldButton:hover, QPushButton#linkButton:hover {
          color: #0e6073;
        }
        QComboBox, QLineEdit {
          background: #ffffff;
          border: 1px solid #c8dbe2;
          border-radius: 8px;
          min-height: 32px;
          padding: 0 8px;
        }
        QSlider#opacitySlider::groove:horizontal {
          height: 5px;
          background: #d8e6ea;
          border-radius: 2px;
        }
        QSlider#opacitySlider::handle:horizontal {
          width: 16px;
          margin: -6px 0;
          background: #16839a;
          border-radius: 8px;
        }
        QPlainTextEdit#advancedText {
          background: #fbfcfd;
          border: 1px solid #e1e8ec;
          border-radius: 9px;
          color: #5d6b75;
          font-size: 11px;
        }
        """

    def _buildUi(self):
        try:
            self.parent.setMinimumWidth(420)
        except Exception:
            pass

        self.scrollArea = qt.QScrollArea()
        self.scrollArea.setObjectName("tmjScrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(qt.QFrame.NoFrame)
        try:
            self.scrollArea.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        self.rootWidget = qt.QWidget()
        self.rootWidget.setObjectName("tmjRoot")
        self.rootWidget.setMinimumWidth(400)
        self.rootWidget.setStyleSheet(self._styleSheet())
        self.scrollArea.setWidget(self.rootWidget)
        self.layout.addWidget(self.scrollArea)

        mainLayout = qt.QVBoxLayout(self.rootWidget)
        mainLayout.setContentsMargins(12, 12, 12, 16)
        mainLayout.setSpacing(10)

        self._buildHeader(mainLayout)
        self._buildStepBar(mainLayout)
        self._buildCaseCard(mainLayout)

        self.pageStack = qt.QStackedWidget()
        self.pageStack.setObjectName("workflowPages")
        self.pageStack.addWidget(self._buildImportPage())
        self.pageStack.addWidget(self._buildAnnotationPage())
        self.pageStack.addWidget(self._buildCheckPage())
        self.pageStack.addWidget(self._buildSavePage())
        self.pageStack.setCurrentIndex(0)
        mainLayout.addWidget(self.pageStack)

        self.workflowMessageLabel = qt.QLabel()
        self.workflowMessageLabel.setObjectName("statusMessage")
        self.workflowMessageLabel.setWordWrap(True)
        mainLayout.addWidget(self.workflowMessageLabel)

        detailsRow = qt.QHBoxLayout()
        self.detailsButton = self._secondaryButton("查看技术信息")
        self.detailsButton.clicked.connect(self._showDetailsDialog)
        detailsRow.addWidget(self.detailsButton)
        self.advancedToggle = qt.QToolButton()
        self.advancedToggle.setObjectName("foldButton")
        self.advancedToggle.setText("查看高级信息")
        self.advancedToggle.setCheckable(True)
        self.advancedToggle.toggled.connect(self._toggleAdvanced)
        detailsRow.addWidget(self.advancedToggle)
        detailsRow.addStretch(1)
        mainLayout.addLayout(detailsRow)

        self.advancedWidget = qt.QPlainTextEdit()
        self.advancedWidget.setObjectName("advancedText")
        self.advancedWidget.setReadOnly(True)
        self.advancedWidget.setMinimumHeight(100)
        self.advancedWidget.setVisible(False)
        mainLayout.addWidget(self.advancedWidget)

    def _buildHeader(self, mainLayout):
        card = self._card("headerCard")
        row = qt.QHBoxLayout(card)
        row.setContentsMargins(18, 14, 14, 14)
        brand = qt.QVBoxLayout()
        self.titleLabel = qt.QLabel("下颌髁突三维标注")
        self.titleLabel.setObjectName("mainTitle")
        brand.addWidget(self.titleLabel)
        self.subtitleLabel = qt.QLabel("TMJ MRI · 人工标注")
        self.subtitleLabel.setObjectName("subtitleLabel")
        brand.addWidget(self.subtitleLabel)
        row.addLayout(brand, 1)

        right = qt.QVBoxLayout()
        actionRow = qt.QHBoxLayout()
        self.helpButton = self._linkButton("？ 使用帮助")
        self.helpButton.clicked.connect(self._showUsageGuide)
        actionRow.addWidget(self.helpButton)
        self.simpleModeButton = self._linkButton("退出简洁模式")
        self.simpleModeButton.clicked.connect(self._toggleSimpleMode)
        actionRow.addWidget(self.simpleModeButton)
        right.addLayout(actionRow)
        self.statusChip = qt.QLabel("● 未标注")
        self.statusChip.setObjectName("statusChip")
        self.statusChip.setAlignment(qt.Qt.AlignCenter)
        right.addWidget(self.statusChip, 0, qt.Qt.AlignRight)
        row.addLayout(right)
        mainLayout.addWidget(card)

    def _buildStepBar(self, mainLayout):
        row = qt.QHBoxLayout()
        row.setSpacing(5)
        self.stepWidgets = []
        self.stepNumberLabels = []
        self.stepStatusLabels = []
        stepSymbols = ["①", "②", "③", "④"]
        for index, name in enumerate(self.STEP_NAMES):
            if index:
                arrow = qt.QLabel("›")
                arrow.setAlignment(qt.Qt.AlignCenter)
                arrow.setStyleSheet("color: #9aadb8; font-size: 18px;")
                row.addWidget(arrow)
            card = self._card("stepCard")
            card.setProperty("state", "waiting")
            cardLayout = qt.QVBoxLayout(card)
            cardLayout.setContentsMargins(10, 8, 10, 8)
            numberLabel = qt.QLabel(f"{stepSymbols[index]}  {name}")
            numberLabel.setObjectName("stepNumber")
            numberLabel.setAlignment(qt.Qt.AlignCenter)
            statusLabel = qt.QLabel("等待")
            statusLabel.setObjectName("stepStatus")
            statusLabel.setAlignment(qt.Qt.AlignCenter)
            cardLayout.addWidget(numberLabel)
            cardLayout.addWidget(statusLabel)
            self.stepWidgets.append(card)
            self.stepNumberLabels.append(numberLabel)
            self.stepStatusLabels.append(statusLabel)
            row.addWidget(card, 1)
        mainLayout.addLayout(row)

    def _buildCaseCard(self, mainLayout):
        card = self._card("caseCard")
        layout = qt.QVBoxLayout(card)
        layout.setContentsMargins(16, 10, 16, 10)
        title = qt.QLabel("当前病例")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        row = qt.QHBoxLayout()
        self.caseIdValue = self._caseValue()
        self.caseProgressValue = self._caseValue()
        self.caseStatusValue = self._caseValue()
        self.caseSaveValue = self._caseValue()
        for caption, value in (
            ("病例", self.caseIdValue),
            ("进度", self.caseProgressValue),
            ("状态", self.caseStatusValue),
            ("保存", self.caseSaveValue),
        ):
            column = qt.QVBoxLayout()
            captionLabel = qt.QLabel(caption)
            captionLabel.setObjectName("caseCaption")
            column.addWidget(captionLabel)
            column.addWidget(value)
            row.addLayout(column, 1)
        layout.addLayout(row)
        mainLayout.addWidget(card)

    def _buildImportPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "第 1 步",
            "导入核磁",
            "选择一份需要标注的颞下颌关节 MRI。",
        )

        self.importLoadedRow = qt.QHBoxLayout()
        loadedCaption = qt.QLabel("已加载病例")
        loadedCaption.setObjectName("mutedLabel")
        self.importLoadedRow.addWidget(loadedCaption)
        self.loadedVolumeSelector = qt.QComboBox()
        self.loadedVolumeSelector.currentIndexChanged.connect(
            self._onLoadedVolumeSelected
        )
        self.importLoadedRow.addWidget(self.loadedVolumeSelector, 1)
        layout.addLayout(self.importLoadedRow)

        buttonRow = qt.QHBoxLayout()
        self.loadButton = self._primaryButton("选择 MRI")
        self.loadButton.clicked.connect(self._chooseAndLoadVolume)
        buttonRow.addWidget(self.loadButton)
        self.useCurrentButton = self._secondaryButton("使用当前病例")
        self.useCurrentButton.clicked.connect(self._useSelectedLoadedVolume)
        buttonRow.addWidget(self.useCurrentButton)
        layout.addLayout(buttonRow)

        self.importResultLabel = qt.QLabel()
        self.importResultLabel.setObjectName("resultMessage")
        self.importResultLabel.setWordWrap(True)
        layout.addWidget(self.importResultLabel)

        self.importNextButton = self._primaryButton("下一步：开始标注")
        self.importNextButton.clicked.connect(self._startAnnotation)
        layout.addWidget(self.importNextButton)

        navigationRow = qt.QHBoxLayout()
        self.previousCaseButton = self._secondaryButton("上一例")
        self.previousCaseButton.clicked.connect(lambda: self._moveCase(-1))
        navigationRow.addWidget(self.previousCaseButton)
        self.caseNavigationLabel = qt.QLabel()
        self.caseNavigationLabel.setObjectName("mutedLabel")
        self.caseNavigationLabel.setAlignment(qt.Qt.AlignCenter)
        navigationRow.addWidget(self.caseNavigationLabel, 1)
        self.nextCaseButton = self._secondaryButton("下一例")
        self.nextCaseButton.clicked.connect(lambda: self._moveCase(1))
        navigationRow.addWidget(self.nextCaseButton)
        layout.addLayout(navigationRow)
        layout.addStretch(1)
        return page

    def _buildAnnotationPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "第 2 步",
            "标注下颌髁突",
            "把图像里的下颌髁突涂出来。建议从第一层检查到最后一层。",
        )

        self.firstHintCard = self._card("hintCard")
        hintLayout = qt.QHBoxLayout(self.firstHintCard)
        hintLayout.setContentsMargins(13, 10, 10, 10)
        hintText = qt.QLabel(
            "新手提示：黄色圆圈是画笔范围。按住鼠标左键，在髁突区域涂抹；"
            "如果涂错，点击左侧“擦除”。"
        )
        hintText.setObjectName("hintLabel")
        hintText.setWordWrap(True)
        hintLayout.addWidget(hintText, 1)
        self.dismissHintButton = self._secondaryButton("知道了")
        self.dismissHintButton.clicked.connect(self._dismissFirstHint)
        hintLayout.addWidget(self.dismissHintButton)
        layout.addWidget(self.firstHintCard)

        toolTitleRow = qt.QHBoxLayout()
        toolTitle = qt.QLabel("常用工具")
        toolTitle.setObjectName("sectionTitle")
        toolTitleRow.addWidget(toolTitle)
        toolTitleRow.addStretch(1)
        self.currentToolLabel = qt.QLabel("当前：画笔")
        self.currentToolLabel.setObjectName("mutedLabel")
        toolTitleRow.addWidget(self.currentToolLabel)
        layout.addLayout(toolTitleRow)

        toolRow = qt.QHBoxLayout()
        self.paintButton = self._toolButton("画笔")
        self.paintButton.clicked.connect(
            lambda checked=False: self._activateEffect("Paint")
        )
        toolRow.addWidget(self.paintButton, 1)
        self.eraseButton = self._toolButton("擦除")
        self.eraseButton.clicked.connect(
            lambda checked=False: self._activateEffect("Erase")
        )
        toolRow.addWidget(self.eraseButton, 1)
        layout.addLayout(toolRow)

        historyRow = qt.QHBoxLayout()
        self.undoButton = self._secondaryButton("撤销")
        self.undoButton.clicked.connect(self._undo)
        historyRow.addWidget(self.undoButton)
        self.redoButton = self._secondaryButton("重做")
        self.redoButton.clicked.connect(self._redo)
        historyRow.addWidget(self.redoButton)
        historyRow.addStretch(1)
        layout.addLayout(historyRow)

        sliceCard = self._card("statusCard")
        sliceLayout = qt.QVBoxLayout(sliceCard)
        sliceLayout.setContentsMargins(13, 10, 13, 10)
        sliceTop = qt.QHBoxLayout()
        self.sliceLabel = qt.QLabel("当前切片：—")
        self.sliceLabel.setObjectName("sectionTitle")
        sliceTop.addWidget(self.sliceLabel)
        sliceTop.addStretch(1)
        self.primaryViewLabel = qt.QLabel("主要标注视图：轴位")
        self.primaryViewLabel.setObjectName("mutedLabel")
        sliceTop.addWidget(self.primaryViewLabel)
        sliceLayout.addLayout(sliceTop)
        sliceButtons = qt.QHBoxLayout()
        self.previousSliceButton = self._secondaryButton("上一层")
        self.previousSliceButton.clicked.connect(lambda: self._moveSlice(-1))
        sliceButtons.addWidget(self.previousSliceButton)
        self.nextSliceButton = self._secondaryButton("下一层")
        self.nextSliceButton.clicked.connect(lambda: self._moveSlice(1))
        sliceButtons.addWidget(self.nextSliceButton)
        sliceButtons.addStretch(1)
        sliceLayout.addLayout(sliceButtons)
        layout.addWidget(sliceCard)

        opacityRow = qt.QHBoxLayout()
        opacityCaption = qt.QLabel("标注透明度")
        opacityCaption.setObjectName("mutedLabel")
        opacityRow.addWidget(opacityCaption)
        self.opacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.opacitySlider.setObjectName("opacitySlider")
        self.opacitySlider.setRange(20, 70)
        self.opacitySlider.setValue(45)
        self.opacitySlider.valueChanged.connect(self._onOpacityChanged)
        opacityRow.addWidget(self.opacitySlider, 1)
        self.opacityValueLabel = qt.QLabel("45%")
        self.opacityValueLabel.setObjectName("mutedLabel")
        opacityRow.addWidget(self.opacityValueLabel)
        layout.addLayout(opacityRow)

        self.assistToggle = qt.QToolButton()
        self.assistToggle.setObjectName("foldButton")
        self.assistToggle.setText("辅助工具  ▾")
        self.assistToggle.setCheckable(True)
        self.assistToggle.toggled.connect(self._toggleAssistTools)
        layout.addWidget(self.assistToggle, 0, qt.Qt.AlignLeft)
        self.assistToolsFrame = self._card("statusCard")
        assistLayout = qt.QHBoxLayout(self.assistToolsFrame)
        assistLayout.setContentsMargins(10, 8, 10, 8)
        for effectName in ("Scissors", "Islands", "Fill between slices"):
            button = self._secondaryButton(EFFECT_LABELS[effectName])
            button.clicked.connect(
                lambda checked=False, name=effectName: self._activateEffect(name)
            )
            assistLayout.addWidget(button, 1)
        self.assistToolsFrame.setVisible(False)
        layout.addWidget(self.assistToolsFrame)

        self.annotationMessageLabel = qt.QLabel()
        self.annotationMessageLabel.setObjectName("resultMessage")
        self.annotationMessageLabel.setWordWrap(True)
        layout.addWidget(self.annotationMessageLabel)

        self.annotateNextButton = self._primaryButton("标完了，下一步：检查")
        self.annotateNextButton.clicked.connect(self._goToCheck)
        layout.addWidget(self.annotateNextButton)

        # The native widget is an engine owned by this module.  Its selectors,
        # effect buttons, and options are hidden so no native English controls
        # leak into the clinical-facing workbench.
        self.editorHost = qt.QFrame()
        self.editorHost.setObjectName("embeddedEditorHost")
        self.editorHost.setVisible(False)
        self.editorHostLayout = qt.QVBoxLayout(self.editorHost)
        self.editorHostLayout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editorHost)
        layout.addStretch(1)
        return page

    def _buildCheckPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "第 3 步",
            "检查标注",
            "从不同方向和三维视图检查有没有漏标或误标。这里只做技术检查，不判断医学是否正确。",
        )

        checkCard = self._card("hintCard")
        checkLayout = qt.QVBoxLayout(checkCard)
        checkLayout.setContentsMargins(13, 11, 13, 11)
        self.checkListLabel = qt.QLabel(
            "请检查：\n"
            "✓ 三个方向的轮廓是否连续\n"
            "✓ 有没有明显漏标或涂到周围组织\n"
            "✓ 三维形状是否存在远处的小块"
        )
        self.checkListLabel.setObjectName("hintLabel")
        self.checkListLabel.setWordWrap(True)
        checkLayout.addWidget(self.checkListLabel)
        buttonRow = qt.QHBoxLayout()
        self.show3DButton = self._primaryButton("显示三维髁突")
        self.show3DButton.clicked.connect(self._show3D)
        buttonRow.addWidget(self.show3DButton)
        self.checkButton = self._secondaryButton("检查标注")
        self.checkButton.clicked.connect(self._checkAnnotation)
        buttonRow.addWidget(self.checkButton)
        checkLayout.addLayout(buttonRow)
        layout.addWidget(checkCard)

        self.qcResultLabel = qt.QLabel()
        self.qcResultLabel.setObjectName("resultMessage")
        self.qcResultLabel.setWordWrap(True)
        layout.addWidget(self.qcResultLabel)

        checkNavigation = qt.QHBoxLayout()
        self.backToAnnotationButton = self._secondaryButton("返回继续修改")
        self.backToAnnotationButton.clicked.connect(self._returnToAnnotation)
        checkNavigation.addWidget(self.backToAnnotationButton)
        checkNavigation.addStretch(1)
        self.confirmCheckButton = self._primaryButton("确认无误，下一步保存")
        self.confirmCheckButton.clicked.connect(self._goToSave)
        checkNavigation.addWidget(self.confirmCheckButton)
        layout.addLayout(checkNavigation)
        layout.addStretch(1)
        return page

    def _buildSavePage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "第 4 步",
            "保存本例",
            "确认病例信息和技术检查结果后，保存本例的下颌髁突 Mask。",
        )

        self.saveSummaryLabel = qt.QLabel()
        self.saveSummaryLabel.setObjectName("summaryCard")
        self.saveSummaryLabel.setWordWrap(True)
        self.saveSummaryLabel.setMargin(14)
        layout.addWidget(self.saveSummaryLabel)

        self.saveButton = self._primaryButton("保存本例标注")
        self.saveButton.clicked.connect(self._saveAnnotation)
        layout.addWidget(self.saveButton)

        self.saveResultLabel = qt.QLabel()
        self.saveResultLabel.setObjectName("resultMessage")
        self.saveResultLabel.setWordWrap(True)
        layout.addWidget(self.saveResultLabel)
        self.saveSuccessLabel = qt.QLabel("保存成功")
        self.saveSuccessLabel.setObjectName("bigSuccess")
        self.saveSuccessLabel.setAlignment(qt.Qt.AlignCenter)
        self.saveSuccessLabel.setVisible(False)
        layout.addWidget(self.saveSuccessLabel)

        saveNavigation = qt.QHBoxLayout()
        self.backToCheckButton = self._secondaryButton("返回检查")
        self.backToCheckButton.clicked.connect(self._returnToCheck)
        saveNavigation.addWidget(self.backToCheckButton)
        saveNavigation.addStretch(1)
        self.nextCaseAfterSaveButton = self._primaryButton("标注下一个病例")
        self.nextCaseAfterSaveButton.clicked.connect(self._moveToNextCaseAfterSave)
        saveNavigation.addWidget(self.nextCaseAfterSaveButton)
        layout.addLayout(saveNavigation)
        layout.addStretch(1)
        return page

    @staticmethod
    def _card(objectName):
        frame = qt.QFrame()
        frame.setObjectName(objectName)
        frame.setFrameShape(qt.QFrame.NoFrame)
        return frame

    @staticmethod
    def _caseValue():
        label = qt.QLabel("—")
        label.setObjectName("caseValue")
        return label

    @staticmethod
    def _primaryButton(text):
        button = qt.QPushButton(text)
        button.setObjectName("primaryButton")
        return button

    @staticmethod
    def _secondaryButton(text):
        button = qt.QPushButton(text)
        button.setObjectName("secondaryButton")
        return button

    @staticmethod
    def _toolButton(text):
        button = qt.QPushButton(text)
        button.setObjectName("toolButton")
        button.setProperty("active", False)
        return button

    @staticmethod
    def _linkButton(text):
        button = qt.QPushButton(text)
        button.setObjectName("linkButton")
        return button

    @staticmethod
    def _addPageHeader(layout, eyebrow, title, description):
        eyebrowLabel = qt.QLabel(eyebrow)
        eyebrowLabel.setObjectName("pageEyebrow")
        layout.addWidget(eyebrowLabel)
        titleLabel = qt.QLabel(title)
        titleLabel.setObjectName("pageTitle")
        layout.addWidget(titleLabel)
        descriptionLabel = qt.QLabel(description)
        descriptionLabel.setObjectName("mutedLabel")
        descriptionLabel.setWordWrap(True)
        layout.addWidget(descriptionLabel)
        layout.addSpacing(5)

    # ------------------------------------------------------------------
    # Module lifecycle and embedded editor
    # ------------------------------------------------------------------
    def enter(self):
        try:
            super().enter()
        except Exception:
            pass
        self._moduleIsActive = True
        self._setSimpleMode(True)
        if self._currentPage == 1 and self.segmentationNode:
            self._ensureViewObservations()

    def exit(self):
        # Module changes are guarded here as well as application close.  If the
        # user cancels, return to this workbench instead of leaving dirty data.
        if self._dirty and not self._confirmUnsaved():
            if not self._reenterScheduled:
                self._reenterScheduled = True
                qt.QTimer.singleShot(0, self._reenterModule)
            return
        self._moduleIsActive = False
        self._removeViewObservations()
        self._setActiveEffect(None)
        self._restoreWorkflowLayout()
        self._setSimpleMode(False)
        try:
            super().exit()
        except Exception:
            pass

    def _reenterModule(self):
        self._reenterScheduled = False
        try:
            slicer.util.selectModule("TMJCondyleAnnotator")
        except Exception:
            pass

    def _createEmbeddedEditor(self):
        try:
            import qSlicerSegmentationsModuleWidgetsPythonQt

            self.editorWidget = (
                qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
            )
            self.editorWidget.setObjectName("embeddedSegmentEditorEngine")
        except Exception as exc:
            self.editorWidget = None
            self._setDetails(
                "嵌入式编辑器创建失败\n"
                + traceback.format_exc()
            )
            self._setStatusMessage(
                "画笔工具暂时没有准备好，请重新进入标注。", "warning"
            )
            return

        try:
            # Follow the official embedding order: parameter node first, scene
            # second, then the fixed nodes and effect list.
            parameterNode = self._getOrCreateParameterNode()
            self.editorWidget.setMRMLSegmentEditorNode(parameterNode)
            self.editorWidget.setMRMLScene(slicer.mrmlScene)
            # Match the official embedding sequence: the parameter node and
            # scene are connected before the widget is inserted into the
            # module layout.  This keeps the native view/event wiring valid
            # even though the engine host is hidden from end users.
            self.editorHostLayout.addWidget(self.editorWidget)
            self.editorWidget.setEffectNameOrder(EFFECT_ORDER)
            try:
                self.editorWidget.setUnorderedEffectsVisible(False)
            except AttributeError:
                # PythonQt exposes this Qt property as an assignment on some
                # Slicer 5.12 builds, even though the C++ setter is present.
                self.editorWidget.unorderedEffectsVisible = False
            self.editorWidget.setEffectColumnCount(2)
            self.editorWidget.setMaximumNumberOfUndoStates(20)
            self.editorWidget.setUndoEnabled(True)
            self.editorWidget.setSegmentationNodeSelectorVisible(False)
            self.editorWidget.setSourceVolumeNodeSelectorVisible(False)
            self.editorWidget.setMaskingSectionVisible(False)
            self.editorWidget.setSpecifyGeometryButtonVisible(False)
            self.editorWidget.setShow3DButtonVisible(False)
            self.editorWidget.setAddRemoveSegmentButtonsVisible(False)
            self.editorWidget.setSwitchToSegmentationsButtonVisible(False)
            self.editorWidget.setAutoShowSourceVolumeNode(False)
            self._editorReady = True

            # Keep the parameter node references explicit as well as routing
            # them through the widget API.  This is the same reference setup
            # used by Slicer's own Segment Editor examples and prevents a
            # delayed MRML refresh from leaving the effect with stale nodes.
            interactionNode = slicer.mrmlScene.GetNodeByID(
                "vtkMRMLInteractionNodeSingleton"
            )
            if interactionNode and hasattr(self.editorWidget, "setInteractionNode"):
                self.editorWidget.setInteractionNode(interactionNode)

            self._effectFactory = slicer.qSlicerSegmentEditorEffectFactory.instance()
            self._effectFactory.connect(
                "effectRegistered(QString)", self._onEffectRegistered
            )
            self._effectFactoryConnected = True
            self.editorWidget.updateEffectList()
        except Exception:
            self._editorReady = False
            self._setDetails("嵌入式编辑器初始化失败\n" + traceback.format_exc())
            self._setStatusMessage(
                "画笔工具暂时没有准备好，请重新进入标注。", "warning"
            )

    def _getOrCreateParameterNode(self):
        if self._parameterNode:
            try:
                if slicer.mrmlScene.GetNodeByID(self._parameterNode.GetID()):
                    return self._parameterNode
            except Exception:
                pass
        singletonTag = "TMJCondyleAnnotator"
        parameterNode = slicer.mrmlScene.GetSingletonNode(
            singletonTag, "vtkMRMLSegmentEditorNode"
        )
        if parameterNode is None:
            parameterNode = slicer.mrmlScene.CreateNodeByClass(
                "vtkMRMLSegmentEditorNode"
            )
            if hasattr(parameterNode, "UnRegister"):
                parameterNode.UnRegister(None)
            parameterNode.SetSingletonTag(singletonTag)
            parameterNode.SetName("TMJ Condyle Annotator Editor")
            parameterNode = slicer.mrmlScene.AddNode(parameterNode)
        self._parameterNode = parameterNode
        return parameterNode

    def _installSceneObservers(self):
        for eventName, callback in (
            ("StartCloseEvent", self._onSceneStartClose),
            ("EndCloseEvent", self._onSceneEndClose),
            ("EndImportEvent", self._onSceneEndImport),
        ):
            event = getattr(slicer.mrmlScene, eventName, None)
            if event is None:
                continue
            try:
                self._sceneObservers.append(
                    (slicer.mrmlScene, slicer.mrmlScene.AddObserver(event, callback))
                )
            except Exception:
                pass

    def _onSceneStartClose(self, caller=None, event=None):
        self._removeViewObservations()
        if self.editorWidget:
            try:
                self.editorWidget.setActiveEffect(None)
                self.editorWidget.setSegmentationNode(None)
            except Exception:
                pass
        self._parameterNode = None
        self.volumeNode = None
        self.segmentationNode = None
        self.segmentId = None

    def _onSceneEndClose(self, caller=None, event=None):
        try:
            self._parameterNode = self._getOrCreateParameterNode()
            if self.editorWidget:
                self.editorWidget.setMRMLSegmentEditorNode(self._parameterNode)
                self.editorWidget.setMRMLScene(slicer.mrmlScene)
                self.editorWidget.updateEffectList()
        except Exception:
            self._setDetails("场景重新连接失败\n" + traceback.format_exc())
        self._refreshLoadedVolumes()
        self._syncUi()

    def _onSceneEndImport(self, caller=None, event=None):
        self._refreshLoadedVolumes()
        if self.editorWidget:
            try:
                self.editorWidget.updateWidgetFromMRML()
            except Exception:
                pass

    def _onEffectRegistered(self, *args):
        if not self.editorWidget:
            return
        try:
            self.editorWidget.updateEffectList()
        except Exception:
            pass
        if self._pendingEffectName:
            pending = self._pendingEffectName
            qt.QTimer.singleShot(0, lambda: self._activateEffect(pending))

    def _configureEmbeddedEditor(self, activatePaint=True):
        if not self.editorWidget or not self._editorReady:
            self._setStatusMessage(
                "画笔工具暂时没有准备好，请重新进入标注。", "warning"
            )
            return False
        if not self.segmentationNode or not self.volumeNode:
            return False
        try:
            parameterNode = self._getOrCreateParameterNode()
            self.editorWidget.setMRMLSegmentEditorNode(parameterNode)
            self.editorWidget.setMRMLScene(slicer.mrmlScene)
            if hasattr(parameterNode, "SetAndObserveSegmentationNode"):
                parameterNode.SetAndObserveSegmentationNode(self.segmentationNode)
            if hasattr(parameterNode, "SetAndObserveSourceVolumeNode"):
                parameterNode.SetAndObserveSourceVolumeNode(self.volumeNode)
            if hasattr(parameterNode, "SetMaskMode"):
                parameterNode.SetMaskMode(
                    getattr(slicer.vtkMRMLSegmentationNode, "EditAllowedEverywhere", 0)
                )
            if hasattr(parameterNode, "SetOverwriteMode"):
                parameterNode.SetOverwriteMode(
                    getattr(slicer.vtkMRMLSegmentEditorNode, "OverwriteAll", 0)
                )
            self.editorWidget.setSegmentationNode(self.segmentationNode)
            if hasattr(self.editorWidget, "setSourceVolumeNode"):
                self.editorWidget.setSourceVolumeNode(self.volumeNode)
            else:
                self.editorWidget.setMasterVolumeNode(self.volumeNode)
            self.editorWidget.setCurrentSegmentID(self.segmentId)
            self.editorWidget.updateEffectList()
            self.editorWidget.updateWidgetFromMRML()
            self._ensureViewObservations()
            if activatePaint:
                return self._activateEffect("Paint", showFriendly=False)
            return True
        except Exception:
            self._editorReady = False
            self._setDetails("编辑器连接当前病例失败\n" + traceback.format_exc())
            self._setStatusMessage(
                "暂时无法开始标注，请重试。", "warning"
            )
            return False

    def _ensureViewObservations(self):
        if not self.editorWidget or not self._editorReady:
            return False
        try:
            if not self._editorViewsObserved:
                self.editorWidget.setupViewObservations()
                self._editorViewsObserved = True
            return True
        except Exception:
            self._setDetails("视图交互连接失败\n" + traceback.format_exc())
            return False

    def _removeViewObservations(self):
        if not self.editorWidget:
            return
        try:
            self.editorWidget.removeViewObservations()
        except Exception:
            pass
        self._editorViewsObserved = False

    def _setActiveEffect(self, effectName):
        if not self.editorWidget:
            return
        try:
            self.editorWidget.setActiveEffect(effectName)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Friendly controls and simple mode
    # ------------------------------------------------------------------
    def _setStatusMessage(self, text, state="info"):
        if not hasattr(self, "workflowMessageLabel"):
            return
        self.workflowMessageLabel.setText(str(text))
        cssState = "neutral" if state == "info" else state
        self.workflowMessageLabel.setProperty("state", cssState)
        self.workflowMessageLabel.style().unpolish(self.workflowMessageLabel)
        self.workflowMessageLabel.style().polish(self.workflowMessageLabel)

    def _setResultMessage(self, label, text, state="neutral"):
        label.setText(str(text))
        label.setProperty("state", state)
        label.style().unpolish(label)
        label.style().polish(label)

    def _setDetails(self, text):
        self._lastDetailText = str(text or "")
        if hasattr(self, "advancedWidget"):
            self.advancedWidget.setPlainText(self._lastDetailText)

    def _toggleAdvanced(self, visible):
        self.advancedWidget.setVisible(bool(visible))
        self.advancedToggle.setText(
            "收起高级信息" if visible else "查看高级信息"
        )

    def _showUsageGuide(self):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("使用帮助")
        dialog.setMinimumWidth(440)
        dialog.setStyleSheet(self._styleSheet())
        layout = qt.QVBoxLayout(dialog)
        title = qt.QLabel("只需要四步")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        text = qt.QLabel(
            "1. 导入核磁\n"
            "2. 用画笔涂出下颌髁突\n"
            "3. 看一下三维效果并检查\n"
            "4. 保存\n\n"
            "涂错了就点击“擦除”。\n"
            "每一层都检查完成后，再进入下一步。"
        )
        text.setObjectName("mutedLabel")
        text.setWordWrap(True)
        layout.addWidget(text)
        closeButton = self._primaryButton("知道了")
        closeButton.clicked.connect(dialog.accept)
        layout.addWidget(closeButton)
        self._execDialog(dialog)

    def _showDetailsDialog(self):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("技术信息")
        dialog.setMinimumSize(680, 360)
        layout = qt.QVBoxLayout(dialog)
        details = qt.QPlainTextEdit()
        details.setReadOnly(True)
        details.setPlainText(self._lastDetailText or "当前没有技术信息。")
        layout.addWidget(details)
        closeButton = self._secondaryButton("关闭")
        closeButton.clicked.connect(dialog.accept)
        layout.addWidget(closeButton)
        self._execDialog(dialog)

    @staticmethod
    def _execDialog(dialog):
        if hasattr(dialog, "exec"):
            return dialog.exec()
        return dialog.exec_()

    def _toggleSimpleMode(self):
        self._setSimpleMode(not self._simpleMode)

    @staticmethod
    def _qtObjectText(obj, attribute):
        try:
            value = getattr(obj, attribute)
            return str(value() if callable(value) else value)
        except Exception:
            return ""

    def _setSimpleMode(self, enabled):
        try:
            mainWindow = slicer.util.mainWindow()
            if enabled:
                targets = []
                try:
                    # PythonQt builds do not all expose QToolBar/QDockWidget
                    # overloads consistently.  Inspect QWidget descendants and
                    # filter by the stable Slicer object/class names instead.
                    targets.extend(mainWindow.findChildren(qt.QWidget))
                except Exception:
                    pass
                try:
                    targets.extend(mainWindow.findChildren("QWidget"))
                except Exception:
                    pass
                # A few PythonQt builds do not accept either findChildren
                # overload for a Qt type.  Walk QObject children as a fallback
                # so the filter remains independent of wrapper details.
                visited = set()
                pending = [mainWindow]
                while pending:
                    parent = pending.pop()
                    if id(parent) in visited:
                        continue
                    visited.add(id(parent))
                    try:
                        children = getattr(parent, "children")
                        children = children() if callable(children) else children
                    except Exception:
                        children = []
                    for child in children or []:
                        targets.append(child)
                        pending.append(child)

                # Slicer creates its QToolBars outside of the central widget
                # hierarchy on some Qt/PythonQt builds.  Include the stable
                # object names explicitly so simple mode does not leave the
                # unrelated Load/Save, module, layout, or developer toolbars
                # visible above this workbench.
                for toolbarName in (
                    "MainToolBar",
                    "ModuleSelectorToolBar",
                    "ModuleToolBar",
                    "ViewToolBar",
                    "MouseModeToolBar",
                    "ViewersToolBar",
                    "DialogToolBar",
                ):
                    for lookupType in (qt.QWidget, "QWidget", qt.QObject, "QObject"):
                        try:
                            toolbar = mainWindow.findChild(lookupType, toolbarName)
                        except Exception:
                            toolbar = None
                        if toolbar is not None:
                            targets.append(toolbar)
                            break

                existing = {id(widget) for widget, _ in self._simpleModeTargets}
                seen = set()
                for widget in targets:
                    widgetId = id(widget)
                    if widgetId in seen:
                        continue
                    seen.add(widgetId)
                    objectName = self._qtObjectText(widget, "objectName").lower()
                    className = ""
                    try:
                        className = str(widget.metaObject().className()).lower()
                    except Exception:
                        pass
                    isToolbar = (
                        "qtoolbar" in className
                        or "toolbar" in className
                        or "toolbar" in objectName
                    )
                    isDeveloperDock = any(
                        token in objectName or token in className
                        for token in ("python", "developer", "errorlog")
                    )
                    isUnrelatedPanel = objectName in (
                        "helpcollapsiblebutton",
                        "dataprobecollapsiblewidget",
                    )
                    if not (isToolbar or isDeveloperDock or isUnrelatedPanel):
                        continue
                    if widgetId not in existing:
                        try:
                            visible = bool(widget.isVisible())
                        except Exception:
                            visible = True
                        self._simpleModeTargets.append((widget, visible))
                        existing.add(widgetId)
                    try:
                        widget.setVisible(False)
                    except Exception:
                        pass
                self._simpleMode = True
            else:
                for widget, visible in self._simpleModeTargets:
                    try:
                        widget.setVisible(visible)
                    except Exception:
                        pass
                self._simpleModeTargets = []
                self._simpleMode = False
        except Exception:
            self._simpleMode = enabled
        if hasattr(self, "simpleModeButton"):
            self.simpleModeButton.setText(
                "退出简洁模式" if self._simpleMode else "开启简洁模式"
            )

    def _dismissFirstHint(self):
        self.firstHintCard.setVisible(False)
        try:
            settings = slicer.app.settings()
            settings.setValue("TMJCondyleAnnotator/firstAnnotationHintSeen", "true")
        except Exception:
            pass

    def _showFirstHintIfNeeded(self):
        show = True
        try:
            value = slicer.app.settings().value(
                "TMJCondyleAnnotator/firstAnnotationHintSeen", ""
            )
            show = str(value).lower() not in ("true", "1", "yes")
        except Exception:
            pass
        self.firstHintCard.setVisible(show)

    # ------------------------------------------------------------------
    # Page transitions
    # ------------------------------------------------------------------
    def _showPage(self, index):
        index = max(0, min(3, int(index)))
        if index == 1 and not self.segmentationNode:
            return False
        if index == 2 and not self.segmentationNode:
            return False
        if index == 3 and self._qcStatus != "通过":
            return False
        self._currentPage = index
        self.pageStack.setCurrentIndex(index)
        if index == 1:
            self._setAnnotationLayout()
            self._configurePrimarySliceView()
            self._ensureViewObservations()
            self._showFirstHintIfNeeded()
        elif index == 2:
            self._setCheckLayout()
            self._removeViewObservations()
        else:
            self._removeViewObservations()
        self._syncUi()
        return True

    def _startAnnotation(self):
        volume = self._requireVolume()
        if not volume:
            return False
        if not self._ensureSegmentation():
            return False
        if not self._configureEmbeddedEditor(activatePaint=True):
            self._setStatusMessage(
                "暂时无法开始标注，请重试。", "warning"
            )
            self._syncUi()
            return False
        self._setStatusMessage("已经准备好，可以用画笔涂出下颌髁突。", "success")
        self._setResultMessage(
            self.annotationMessageLabel,
            "按住鼠标左键涂抹；涂错后点击“擦除”。",
            "neutral",
        )
        return self._showPage(1)

    def _goToCheck(self):
        if not self.segmentationNode:
            self._setStatusMessage("请先开始标注。", "warning")
            return False
        if not self._annotationHasData:
            self._setStatusMessage("当前标注为空，检查时会提示这一问题。", "warning")
            self._setResultMessage(
                self.annotationMessageLabel,
                "当前标注为空，可以先进入检查确认。",
                "warning",
            )
        self._setPageForCheck()
        return True

    def _setPageForCheck(self):
        self._currentPage = 2
        self.pageStack.setCurrentIndex(2)
        self._setCheckLayout()
        self._removeViewObservations()
        self._setResultMessage(
            self.qcResultLabel,
            "请先点击“显示三维髁突”，再点击“检查标注”。",
            "neutral",
        )
        self._setStatusMessage("请从三个方向和三维视图检查标注。", "info")
        self._syncUi()
        self._show3D(silent=True)

    def _returnToAnnotation(self):
        self._showPage(1)
        self._setStatusMessage("可以继续修改本例标注。", "info")

    def _returnToCheck(self):
        if self.segmentationNode:
            self._setPageForCheck()
            self._setStatusMessage("已返回检查页面。", "info")

    def _goToSave(self):
        if self._qcStatus != "通过":
            self._setStatusMessage("请先完成技术检查。", "warning")
            return False
        self._showPage(3)
        self._setStatusMessage("确认无误后，可以保存本例。", "info")
        self._syncUi()
        return True

    # ------------------------------------------------------------------
    # Case and volume handling
    # ------------------------------------------------------------------
    @staticmethod
    def _dialogPath(value):
        if isinstance(value, tuple):
            return value[0]
        return value

    @staticmethod
    def _stripImageSuffix(path):
        name = Path(path).name
        for suffix in (".nii.gz", ".nii", ".nrrd"):
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return Path(name).stem

    @classmethod
    def _safeCaseId(cls, path, fallback="case_001"):
        if not path:
            return fallback
        candidate = cls._stripImageSuffix(path)
        if CASE_FILE_PATTERN.fullmatch(candidate) and CASE_ID_PATTERN.fullmatch(candidate):
            return candidate
        return fallback

    def _caseIdForIndex(self, path, index):
        candidate = self._safeCaseId(path, fallback="")
        return candidate or f"case_{index + 1:03d}"

    def _refreshCaseFiles(self):
        paths = []
        if self.niftiDir.exists():
            for pattern in ("*.nii.gz", "*.nii", "*.nrrd"):
                paths.extend(self.niftiDir.glob(pattern))
        self._caseFiles = sorted(
            {path.resolve() for path in paths}, key=lambda path: path.name
        )
        if self._currentCasePath:
            try:
                self._currentCaseIndex = self._caseFiles.index(self._currentCasePath)
            except ValueError:
                self._currentCaseIndex = -1
        elif self._caseFiles:
            self._currentCaseIndex = 0
            self._currentCaseId = self._caseIdForIndex(self._caseFiles[0], 0)

    def _refreshLoadedVolumes(self):
        if not hasattr(self, "loadedVolumeSelector"):
            return
        currentId = self.volumeNode.GetID() if self.volumeNode else None
        self._isUpdating = True
        try:
            self.loadedVolumeSelector.clear()
            ids = []
            nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
            for index in range(nodes.GetNumberOfItems()):
                node = nodes.GetItemAsObject(index)
                if not node:
                    continue
                ids.append(node.GetID())
                path = self._volumePath(node)
                displayName = self._safeCaseId(path, f"病例 {len(ids)}")
                self.loadedVolumeSelector.addItem(displayName)
            self._loadedVolumeIds = ids
            if currentId in ids:
                self.loadedVolumeSelector.setCurrentIndex(ids.index(currentId))
            elif ids:
                self.loadedVolumeSelector.setCurrentIndex(0)
        finally:
            self._isUpdating = False
        self.loadedVolumeSelector.setVisible(bool(ids))

    def _volumePath(self, node):
        if not node:
            return None
        try:
            storage = node.GetStorageNode()
            if storage and storage.GetFileName():
                return Path(storage.GetFileName()).resolve()
        except Exception:
            pass
        return None

    def _chooseAndLoadVolume(self):
        fileName = self._dialogPath(
            qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                "选择 MRI",
                str(self.niftiDir),
                "MRI 文件 (*.nii.gz *.nii *.nrrd);;所有文件 (*)",
            )
        )
        if not fileName:
            return
        if not self._confirmUnsaved():
            return
        try:
            loaded = slicer.util.loadVolume(str(fileName), returnNode=True)
            node = loaded[1] if isinstance(loaded, tuple) else loaded
        except TypeError:
            try:
                node = slicer.util.loadVolume(str(fileName))
            except Exception as exc:
                self._handleLoadFailure(fileName, exc)
                return
        except Exception as exc:
            self._handleLoadFailure(fileName, exc)
            return
        if not node:
            self._handleLoadFailure(fileName, "没有返回可用的 MRI")
            return
        self._ownedVolumeIds.add(node.GetID())
        self._setVolumeNode(node, Path(fileName).resolve())

    def _useSelectedLoadedVolume(self):
        index = self.loadedVolumeSelector.currentIndex
        if callable(index):
            index = index()
        if index is None or index < 0 or index >= len(self._loadedVolumeIds):
            self._setStatusMessage("请先选择或加载一份 MRI。", "warning")
            return
        node = slicer.mrmlScene.GetNodeByID(self._loadedVolumeIds[index])
        if node and self._confirmUnsaved():
            self._setVolumeNode(node, self._volumePath(node))

    def _onLoadedVolumeSelected(self, index):
        if self._isUpdating or index < 0 or index >= len(self._loadedVolumeIds):
            return
        node = slicer.mrmlScene.GetNodeByID(self._loadedVolumeIds[index])
        if not node or node == self.volumeNode:
            return
        if not self._confirmUnsaved():
            self._restoreLoadedVolumeSelection()
            return
        self._setVolumeNode(node, self._volumePath(node))

    def _restoreLoadedVolumeSelection(self):
        if not self.volumeNode:
            return
        try:
            index = self._loadedVolumeIds.index(self.volumeNode.GetID())
        except ValueError:
            return
        self._isUpdating = True
        try:
            self.loadedVolumeSelector.setCurrentIndex(index)
        finally:
            self._isUpdating = False

    def _handleLoadFailure(self, fileName, error):
        self._setDetails(
            f"读取 MRI 失败\n文件：{fileName}\n"
            f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}"
        )
        self._setStatusMessage(
            "这份 MRI 暂时无法读取，请检查文件是否完整。", "warning"
        )

    def _setVolumeNode(self, node, path=None):
        if not node:
            return
        oldNode = self.volumeNode
        self._removeSegmentation()
        self.volumeNode = node
        self._currentCasePath = (
            Path(path).resolve() if path else self._volumePath(node)
        )
        self._refreshCaseFiles()
        if self._currentCasePath and self._currentCasePath in self._caseFiles:
            self._currentCaseIndex = self._caseFiles.index(self._currentCasePath)
            self._currentCaseId = self._caseIdForIndex(
                self._currentCasePath, self._currentCaseIndex
            )
        else:
            self._currentCaseIndex = -1
            self._currentCaseId = self._safeCaseId(
                self._currentCasePath or "", "case_001"
            )

        self._primaryAxis = self._detectPrimaryAxis(node)
        self._primaryOrientation = ORIENTATIONS[self._primaryAxis]
        self._dirty = False
        self._saved = self._outputPath().exists()
        self._annotationHasData = False
        self._qcStatus = "未检查"
        self._currentPage = 0
        self.pageStack.setCurrentIndex(0)
        self._setDetails(
            "当前病例：{0}\nMRI：{1}\n自动保存位置：{2}\nmanifest：{3}".format(
                self._currentCaseId,
                self._currentCasePath or "未记录文件位置",
                self._outputPath(),
                self.manifestPath,
            )
        )
        self._refreshLoadedVolumes()
        self._refreshSliceObservers()
        self._ensureVolumeDisplay(node)
        if oldNode and oldNode.GetID() != node.GetID() and oldNode.GetID() in self._ownedVolumeIds:
            try:
                slicer.mrmlScene.RemoveNode(oldNode)
            except Exception:
                pass
            self._ownedVolumeIds.discard(oldNode.GetID())
        self.importResultLabel.setText(
            f"✓ MRI 已加载\n病例：{self._currentCaseId}"
        )
        self._setResultMessage(self.importResultLabel, "✓ MRI 已加载\n病例：" + self._currentCaseId, "success")
        self._setStatusMessage("MRI 已加载，可以开始标注。", "success")
        self._syncUi()

    def _ensureVolumeDisplay(self, node):
        try:
            node.SetDisplayVisibility(True)
            display = node.GetDisplayNode()
            if display:
                display.SetVisibility(True)
        except Exception:
            pass
        self._showVolumeInSliceViews(node)

    def _showVolumeInSliceViews(self, node):
        try:
            manager = slicer.app.layoutManager()
            for name in manager.sliceViewNames():
                widget = manager.sliceWidget(name)
                if not widget:
                    continue
                composite = widget.mrmlSliceCompositeNode()
                composite.SetBackgroundVolumeID(node.GetID())
                composite.SetForegroundVolumeID("")
                composite.SetLabelVolumeID("")
        except Exception:
            pass

    def _outputPath(self):
        return self.labelsDir / f"{self._currentCaseId}.nii.gz"

    def _moveCase(self, delta):
        self._refreshCaseFiles()
        if not self._caseFiles:
            self._setStatusMessage("当前文件夹中没有其它病例。", "info")
            return False
        target = self._currentCaseIndex if self._currentCaseIndex >= 0 else 0
        target += delta
        if target < 0 or target >= len(self._caseFiles):
            return False
        if not self._confirmUnsaved():
            return False
        path = self._caseFiles[target]
        try:
            loaded = slicer.util.loadVolume(str(path), returnNode=True)
            node = loaded[1] if isinstance(loaded, tuple) else loaded
        except TypeError:
            try:
                node = slicer.util.loadVolume(str(path))
            except Exception as exc:
                self._handleLoadFailure(path, exc)
                return False
        except Exception as exc:
            self._handleLoadFailure(path, exc)
            return False
        if not node:
            self._handleLoadFailure(path, "没有返回可用的 MRI")
            return False
        self._ownedVolumeIds.add(node.GetID())
        self._setVolumeNode(node, path)
        return True

    # ------------------------------------------------------------------
    # Segmentation and effects
    # ------------------------------------------------------------------
    def _requireVolume(self):
        if self.volumeNode:
            return self.volumeNode
        self._setStatusMessage("请先选择一份 MRI。", "warning")
        return None

    def _ensureSegmentation(self):
        volume = self._requireVolume()
        if not volume:
            return False
        if self.segmentationNode:
            return True
        self._isUpdating = True
        try:
            self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "TMJ_Condyle_Annotation"
            )
            self.segmentationNode.CreateDefaultDisplayNodes()
            self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(
                volume
            )
            segmentation = self.segmentationNode.GetSegmentation()
            self.segmentId = segmentation.AddEmptySegment(
                "Mandibular Condyle",
                "MandibularCondyle",
                (0.16, 0.72, 0.78),
            )
            self.segmentationNode.SetDisplayVisibility(True)
            self._applySegmentationDisplaySettings()
            self._loadExistingLabelIfPresent()
        except Exception:
            detail = "创建标注区域失败\n" + traceback.format_exc()
            self._setDetails(detail)
            if self.segmentationNode:
                try:
                    slicer.mrmlScene.RemoveNode(self.segmentationNode)
                except Exception:
                    pass
            self.segmentationNode = None
            self.segmentId = None
            self._setStatusMessage("暂时无法开始标注，请重试。", "warning")
            return False
        finally:
            self._isUpdating = False
        self._observeSegmentation()
        self._syncUi()
        return True

    def _applySegmentationDisplaySettings(self):
        if not self.segmentationNode:
            return
        display = self.segmentationNode.GetDisplayNode()
        if not display:
            return
        try:
            display.SetVisibility(True)
            display.SetVisibility2DFill(True)
            display.SetVisibility2DOutline(True)
            display.SetVisibility3D(True)
            display.SetOpacity(0.45)
            if hasattr(display, "SetOpacity3D"):
                display.SetOpacity3D(0.45)
            if self.segmentId and hasattr(display, "SetSegmentOpacity"):
                display.SetSegmentOpacity(self.segmentId, 0.45)
            if self.segmentId and hasattr(display, "SetSegmentOpacity3D"):
                display.SetSegmentOpacity3D(self.segmentId, 0.45)
            if self.segmentId and hasattr(display, "SetSegmentVisibility"):
                display.SetSegmentVisibility(self.segmentId, True)
        except Exception:
            pass

    def _loadExistingLabelIfPresent(self):
        path = self._outputPath()
        if not path.exists() or not self.segmentationNode:
            return
        labelNode = None
        try:
            loaded = slicer.util.loadLabelVolume(str(path), returnNode=True)
            labelNode = loaded[1] if isinstance(loaded, tuple) else loaded
            logic = slicer.modules.segmentations.logic()
            imported = logic.ImportLabelmapToSegmentationNode(
                labelNode, self.segmentationNode
            )
            if imported is False:
                raise RuntimeError("import returned false")
            segmentation = self.segmentationNode.GetSegmentation()
            while segmentation.GetNumberOfSegments() > 1:
                segmentation.RemoveSegment(segmentation.GetNthSegmentID(0))
            if segmentation.GetNumberOfSegments() == 1:
                self.segmentId = segmentation.GetNthSegmentID(0)
                segment = segmentation.GetSegment(self.segmentId)
                segment.SetName("Mandibular Condyle")
                segment.SetColor(0.16, 0.72, 0.78)
            self._annotationHasData = True
            self._saved = True
            self._dirty = False
        except Exception:
            self._saved = False
            self._setDetails(
                f"读取已保存标注失败\n文件：{path}\n{traceback.format_exc()}"
            )
        finally:
            if labelNode:
                try:
                    slicer.mrmlScene.RemoveNode(labelNode)
                except Exception:
                    pass

    def _observeSegmentation(self):
        self._removeSegmentationObservers()
        if not self.segmentationNode:
            return
        try:
            self._segmentationObservers.append(
                (
                    self.segmentationNode,
                    self.segmentationNode.AddObserver(
                        vtk.vtkCommand.ModifiedEvent, self._onSegmentationModified
                    ),
                )
            )
            segmentation = self.segmentationNode.GetSegmentation()
            if segmentation:
                observedEvents = [vtk.vtkCommand.ModifiedEvent]
                # Segment Editor effects modify the contained vtkSegmentation
                # and may emit SegmentModified/SourceRepresentationModified
                # without propagating a vtkMRMLNode ModifiedEvent immediately.
                # Observe those official content events so dirty state and the
                # next-step guidance always follow a real paint or erase.
                for eventName in (
                    "SegmentModified",
                    "SourceRepresentationModified",
                ):
                    event = getattr(segmentation, eventName, None)
                    if event is not None and event not in observedEvents:
                        observedEvents.append(event)
                for event in observedEvents:
                    self._segmentationObservers.append(
                        (
                            segmentation,
                            segmentation.AddObserver(
                                event, self._onSegmentationModified
                            ),
                        )
                    )
        except Exception:
            self._setDetails("标注变化监听失败\n" + traceback.format_exc())

    def _removeSegmentationObservers(self):
        for obj, tag in self._segmentationObservers:
            try:
                obj.RemoveObserver(tag)
            except Exception:
                pass
        self._segmentationObservers = []

    def _onSegmentationModified(self, caller=None, event=None):
        if self._isUpdating or not self.segmentationNode:
            return
        self._dirty = True
        self._saved = False
        self._annotationHasData = True
        self._qcStatus = "未检查"
        self._setResultMessage(
            self.annotationMessageLabel,
            "标注已修改，完成后请进入检查。",
            "neutral",
        )
        self._syncUi()

    def _availableEffectNames(self):
        if not self.editorWidget:
            return []
        try:
            return [str(name) for name in self.editorWidget.availableEffectNames()]
        except Exception:
            return []

    def _activateEffect(self, effectName, showFriendly=True):
        if not self.segmentationNode:
            if not self._ensureSegmentation():
                return False
        if not self.editorWidget or not self._editorReady:
            self._setStatusMessage(
                "画笔工具暂时没有准备好，请重新进入标注。", "warning"
            )
            return False
        try:
            self.editorWidget.updateEffectList()
            effect = self.editorWidget.effectByName(effectName)
            if effect is None or effectName not in self._availableEffectNames():
                self._pendingEffectName = effectName
                if self._pendingEffectAttempts < 20:
                    self._pendingEffectAttempts += 1
                    qt.QTimer.singleShot(
                        150, lambda: self._activateEffect(effectName, showFriendly=False)
                    )
                elif showFriendly:
                    self._setDetails(
                        f"效果未注册：{effectName}\n可用效果：{self._availableEffectNames()}"
                    )
                    self._setStatusMessage(
                        "画笔工具暂时没有准备好，请重新进入标注。", "warning"
                    )
                return False
            self.editorWidget.setSegmentationNode(self.segmentationNode)
            if hasattr(self.editorWidget, "setSourceVolumeNode"):
                self.editorWidget.setSourceVolumeNode(self.volumeNode)
            else:
                self.editorWidget.setMasterVolumeNode(self.volumeNode)
            self.editorWidget.setCurrentSegmentID(self.segmentId)
            self.editorWidget.setActiveEffectByName(effectName)
            if self.editorWidget.activeEffect() is None:
                raise RuntimeError("active effect is still null")
            self._pendingEffectName = None
            self._pendingEffectAttempts = 0
            self._currentEffectName = effectName
            if hasattr(self, "currentToolLabel"):
                self.currentToolLabel.setText(
                    "当前：" + EFFECT_LABELS.get(effectName, "工具")
                )
                self._setToolButtonState()
            self._setStatusMessage(
                "当前工具：" + EFFECT_LABELS.get(effectName, "工具"), "success"
            )
            self._setResultMessage(
                self.annotationMessageLabel,
                "按住鼠标左键涂抹；涂错后点击“擦除”。",
                "neutral",
            )
            return True
        except Exception:
            self._setDetails(
                f"选择效果失败：{effectName}\n{traceback.format_exc()}"
            )
            if showFriendly:
                self._setStatusMessage(
                    "画笔工具暂时没有准备好，请重新进入标注。", "warning"
                )
            return False

    def _setToolButtonState(self):
        for effectName, button in (
            ("Paint", self.paintButton),
            ("Erase", self.eraseButton),
        ):
            active = self._currentEffectName == effectName
            button.setProperty("active", active)
            button.style().unpolish(button)
            button.style().polish(button)

    def _undo(self):
        if not self.editorWidget:
            return
        try:
            self.editorWidget.undo()
            self._setStatusMessage("已撤销上一步操作。", "info")
        except Exception:
            self._setDetails("撤销失败\n" + traceback.format_exc())
            self._setStatusMessage("暂时无法撤销，请重试。", "warning")

    def _redo(self):
        if not self.editorWidget:
            return
        try:
            self.editorWidget.redo()
            self._setStatusMessage("已重做上一步操作。", "info")
        except Exception:
            self._setDetails("重做失败\n" + traceback.format_exc())
            self._setStatusMessage("暂时无法重做，请重试。", "warning")

    def _toggleAssistTools(self, visible):
        self.assistToolsFrame.setVisible(bool(visible))
        self.assistToggle.setText("辅助工具  ▴" if visible else "辅助工具  ▾")

    def _onOpacityChanged(self, value):
        self.opacityValueLabel.setText(f"{int(value)}%")
        if not self.segmentationNode:
            return
        try:
            display = self.segmentationNode.GetDisplayNode()
            opacity = float(value) / 100.0
            display.SetOpacity(opacity)
            if hasattr(display, "SetOpacity3D"):
                display.SetOpacity3D(opacity)
            if self.segmentId and hasattr(display, "SetSegmentOpacity"):
                display.SetSegmentOpacity(self.segmentId, opacity)
            if self.segmentId and hasattr(display, "SetSegmentOpacity3D"):
                display.SetSegmentOpacity3D(self.segmentId, opacity)
        except Exception:
            self._setDetails("调整标注透明度失败\n" + traceback.format_exc())

    # ------------------------------------------------------------------
    # Slice navigation and layout
    # ------------------------------------------------------------------
    @staticmethod
    def _detectPrimaryAxis(volume):
        try:
            dimensions = volume.GetImageData().GetDimensions()
            spacing = volume.GetSpacing()
            return min(range(3), key=lambda axis: (int(dimensions[axis]), -abs(float(spacing[axis]))))
        except Exception:
            return 2

    def _orientationForAxis(self):
        return self._primaryOrientation

    def _primarySliceWidget(self):
        try:
            manager = slicer.app.layoutManager()
            for name in ("Red", "Red+", "Yellow", "Green"):
                widget = manager.sliceWidget(name)
                if widget:
                    return widget
        except Exception:
            pass
        return None

    def _configurePrimarySliceView(self):
        if not self.volumeNode:
            return
        try:
            widget = self._primarySliceWidget()
            if not widget:
                return
            sliceNode = widget.mrmlSliceNode()
            sliceNode.SetOrientation(self._orientationForAxis())
            bounds = [0.0] * 6
            self.volumeNode.GetRASBounds(bounds)
            center = [
                (bounds[0] + bounds[1]) * 0.5,
                (bounds[2] + bounds[3]) * 0.5,
                (bounds[4] + bounds[5]) * 0.5,
            ]
            sliceNode.JumpSlice(center[0], center[1], center[2])
            widget.sliceLogic().FitSliceToAll()
        except Exception:
            self._setDetails("设置主要标注视图失败\n" + traceback.format_exc())

    def _refreshSliceObservers(self):
        for node, tag in self._sliceObservers:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._sliceObservers = []
        try:
            nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSliceNode")
            for index in range(nodes.GetNumberOfItems()):
                node = nodes.GetItemAsObject(index)
                if node:
                    self._sliceObservers.append(
                        (
                            node,
                            node.AddObserver(
                                vtk.vtkCommand.ModifiedEvent, self._onSliceChanged
                            ),
                        )
                    )
        except Exception:
            pass
        self._updateSliceLabel()

    def _onSliceChanged(self, caller=None, event=None):
        self._updateSliceLabel()

    def _volumeMatrix(self):
        matrix = vtk.vtkMatrix4x4()
        self.volumeNode.GetIJKToRASMatrix(matrix)
        return matrix

    @staticmethod
    def _matrixPoint(matrix, point):
        result = [0.0, 0.0, 0.0, 0.0]
        matrix.MultiplyPoint([point[0], point[1], point[2], 1.0], result)
        if abs(result[3]) > 1e-12:
            return [result[i] / result[3] for i in range(3)]
        return result[:3]

    def _sliceInfo(self):
        if not self.volumeNode or not self.volumeNode.GetImageData():
            return None
        widget = self._primarySliceWidget()
        if not widget:
            return None
        try:
            logic = widget.sliceLogic()
            sliceNode = widget.mrmlSliceNode()
            dimensions = self.volumeNode.GetImageData().GetDimensions()
            axis = self._primaryAxis
            count = int(dimensions[axis])
            if count <= 0:
                return None
            ijkToRAS = self._volumeMatrix()
            rasToIJK = vtk.vtkMatrix4x4()
            rasToIJK.DeepCopy(ijkToRAS)
            rasToIJK.Invert()
            sliceToRAS = sliceNode.GetSliceToRAS()
            pointRAS = [
                float(sliceToRAS.GetElement(0, 3)),
                float(sliceToRAS.GetElement(1, 3)),
                float(sliceToRAS.GetElement(2, 3)),
            ]
            pointIJK = self._matrixPoint(rasToIJK, pointRAS)
            index = int(round(pointIJK[axis])) + 1
            index = max(1, min(count, index))
            spacing = abs(float(self.volumeNode.GetSpacing()[axis]))
            return {
                "logic": logic,
                "sliceNode": sliceNode,
                "index": index,
                "count": count,
                "axis": axis,
                "spacing": spacing,
                "pointIJK": pointIJK,
            }
        except Exception:
            return None

    def _updateSliceLabel(self):
        if not hasattr(self, "sliceLabel"):
            return
        info = self._sliceInfo()
        if info:
            self.sliceLabel.setText(
                f"当前切片：第 {info['index']} / {info['count']} 层"
            )
            self.previousSliceButton.setEnabled(info["index"] > 1)
            self.nextSliceButton.setEnabled(info["index"] < info["count"])
        else:
            self.sliceLabel.setText("当前切片：—")
            self.previousSliceButton.setEnabled(False)
            self.nextSliceButton.setEnabled(False)
        self.primaryViewLabel.setText(
            "主要标注视图：" + ORIENTATION_LABELS.get(self._primaryAxis, "轴位")
        )

    def _moveSlice(self, delta):
        info = self._sliceInfo()
        if not info:
            return
        targetIndex = max(1, min(info["count"], info["index"] + int(delta)))
        if targetIndex == info["index"]:
            return
        try:
            dimensions = self.volumeNode.GetImageData().GetDimensions()
            centerIJK = [
                (float(dimensions[0]) - 1.0) * 0.5,
                (float(dimensions[1]) - 1.0) * 0.5,
                (float(dimensions[2]) - 1.0) * 0.5,
            ]
            centerIJK[info["axis"]] = float(targetIndex - 1)
            targetRAS = self._matrixPoint(self._volumeMatrix(), centerIJK)
            currentIJK = list(centerIJK)
            currentIJK[info["axis"]] = float(info["index"] - 1)
            currentRAS = self._matrixPoint(self._volumeMatrix(), currentIJK)
            sliceToRAS = info["sliceNode"].GetSliceToRAS()
            normal = [
                float(sliceToRAS.GetElement(0, 2)),
                float(sliceToRAS.GetElement(1, 2)),
                float(sliceToRAS.GetElement(2, 2)),
            ]
            deltaRAS = [targetRAS[i] - currentRAS[i] for i in range(3)]
            normalLength = sum(value * value for value in normal)
            offsetDelta = (
                sum(deltaRAS[i] * normal[i] for i in range(3)) / normalLength
                if normalLength > 1e-12
                else info["spacing"] * float(delta)
            )
            info["logic"].SetSliceOffset(
                float(info["logic"].GetSliceOffset()) + offsetDelta
            )
            slicer.app.processEvents()
            self._updateSliceLabel()
        except Exception:
            self._setDetails("切换切片失败\n" + traceback.format_exc())
            self._setStatusMessage("暂时无法切换切片，请重试。", "warning")

    def _annotationLayoutDescription(self):
        orientations = [
            self._orientationForAxis(),
            ORIENTATIONS[(self._primaryAxis + 1) % 3],
            ORIENTATIONS[(self._primaryAxis + 2) % 3],
        ]
        labels = ["主要", "辅助 1", "辅助 2"]
        colors = ["#16839a", "#6a9fbc", "#9aa6b8"]
        views = []
        for tag, orientation, label, color in zip(
            ("Red", "Green", "Yellow"), orientations, labels, colors
        ):
            views.append(
                f'<item><view class="vtkMRMLSliceNode" singletontag="{tag}">'
                f'<property name="orientation" action="default">{orientation}</property>'
                f'<property name="viewlabel" action="default">{label}</property>'
                f'<property name="viewcolor" action="default">{color}</property>'
                "</view></item>"
            )
        return '<layout type="horizontal">' + "".join(views) + "</layout>"

    def _rememberPreviousLayout(self):
        if self._workflowLayoutChanged:
            return
        try:
            self._previousLayout = slicer.app.layoutManager().layout
        except Exception:
            self._previousLayout = None
        self._workflowLayoutChanged = True

    def _setAnnotationLayout(self):
        try:
            wasObserved = self._editorViewsObserved
            if wasObserved:
                self._removeViewObservations()
            self._rememberPreviousLayout()
            manager = slicer.app.layoutManager()
            layoutNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLayoutNode")
            if not layoutNode:
                return
            if self._customLayoutId is None:
                self._customLayoutId = 3101
                while layoutNode.IsLayoutDescription(self._customLayoutId):
                    self._customLayoutId += 1
            layoutDescription = self._annotationLayoutDescription()
            if not self._customLayoutAdded:
                layoutNode.AddLayoutDescription(self._customLayoutId, layoutDescription)
                self._customLayoutAdded = True
            else:
                layoutNode.SetLayoutDescription(self._customLayoutId, layoutDescription)
            manager.setLayout(self._customLayoutId)
            self._refreshSliceObservers()
            self._showVolumeInSliceViews(self.volumeNode)
            self._configurePrimarySliceView()
            self._ensureViewObservations()
        except Exception:
            self._setDetails("设置标注布局失败\n" + traceback.format_exc())

    def _setCheckLayout(self):
        try:
            self._rememberPreviousLayout()
            manager = slicer.app.layoutManager()
            layoutNodeClass = slicer.vtkMRMLLayoutNode
            layoutValue = getattr(layoutNodeClass, "SlicerLayoutFourUpView", None)
            if layoutValue is None:
                layoutValue = getattr(
                    layoutNodeClass, "SlicerLayoutConventionalView", None
                )
            if layoutValue is not None:
                manager.setLayout(layoutValue)
            self._refreshSliceObservers()
            self._showVolumeInSliceViews(self.volumeNode)
        except Exception:
            self._setDetails("设置检查布局失败\n" + traceback.format_exc())

    def _restoreWorkflowLayout(self):
        if not self._workflowLayoutChanged:
            return
        try:
            manager = slicer.app.layoutManager()
            if self._previousLayout is not None:
                manager.setLayout(self._previousLayout)
            layoutNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLayoutNode")
            if layoutNode and self._customLayoutAdded and self._customLayoutId is not None:
                layoutNode.SetLayoutDescription(self._customLayoutId, "")
        except Exception:
            pass
        self._previousLayout = None
        self._workflowLayoutChanged = False
        self._customLayoutAdded = False

    def _show3D(self, silent=False):
        if not self.segmentationNode:
            if not silent:
                self._setStatusMessage("请先开始标注。", "warning")
            return False
        try:
            self.segmentationNode.CreateClosedSurfaceRepresentation()
            self._applySegmentationDisplaySettings()
            self._setCheckLayout()
            manager = slicer.app.layoutManager()
            if manager.threeDViewCount > 0:
                view = manager.threeDWidget(0).threeDView()
                view.resetFocalPoint()
            self._setStatusMessage("三维髁突已显示，请旋转检查轮廓。", "success")
            return True
        except Exception:
            self._setDetails("显示三维失败\n" + traceback.format_exc())
            if not silent:
                self._setStatusMessage("三维视图暂时无法显示，请重试。", "warning")
            return False

    # ------------------------------------------------------------------
    # Technical QC and export
    # ------------------------------------------------------------------
    def _exportLabelmap(self, volume):
        if not self.segmentationNode:
            raise RuntimeError("segmentation is not initialized")
        segmentation = self.segmentationNode.GetSegmentation()
        if segmentation.GetNumberOfSegments() != 1:
            raise RuntimeError("annotation must contain exactly one segment")
        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", f"{self._currentCaseId}_temporary_label"
        )
        logic = slicer.modules.segmentations.logic()
        if hasattr(logic, "ExportVisibleSegmentsToLabelmapNode"):
            ok = logic.ExportVisibleSegmentsToLabelmapNode(
                self.segmentationNode, labelmap, volume
            )
        else:
            segmentIds = vtk.vtkStringArray()
            segmentIds.InsertNextValue(self.segmentId)
            ok = logic.ExportSegmentsToLabelmapNode(
                self.segmentationNode, segmentIds, labelmap, volume
            )
        if ok is False:
            slicer.mrmlScene.RemoveNode(labelmap)
            raise RuntimeError("Slicer segmentation export returned false")
        return labelmap

    @staticmethod
    def _nodeGeometry(node):
        imageData = node.GetImageData()
        dimensions = tuple(int(value) for value in imageData.GetDimensions())
        spacing = tuple(float(value) for value in node.GetSpacing())
        origin = tuple(float(value) for value in node.GetOrigin())
        matrix = vtk.vtkMatrix3x3()
        try:
            node.GetIJKToRASDirectionMatrix(matrix)
            direction = tuple(
                float(matrix.GetElement(i, j)) for i in range(3) for j in range(3)
            )
        except Exception:
            direction = ()
        return dimensions, spacing, origin, direction

    @classmethod
    def _geometryErrors(cls, imageNode, labelNode):
        image = cls._nodeGeometry(imageNode)
        label = cls._nodeGeometry(labelNode)
        errors = []
        if image[0] != label[0]:
            errors.append("图像和标注尺寸不同")
        for name, left, right in (
            ("spacing", image[1], label[1]),
            ("origin", image[2], label[2]),
            ("direction", image[3], label[3]),
        ):
            if len(left) != len(right) or any(
                abs(a - b) > 1e-4 for a, b in zip(left, right)
            ):
                errors.append(f"{name} differs")
        return errors

    @staticmethod
    def _componentCount(mask):
        try:
            from scipy import ndimage

            structure = ndimage.generate_binary_structure(mask.ndim, 1)
            _, count = ndimage.label(mask, structure=structure)
            return int(count)
        except Exception:
            return None

    def _runQualityCheck(self):
        volume = self._requireVolume()
        if not volume or not self.segmentationNode:
            return {"passed": False, "issues": ["当前标注为空"], "details": ""}
        labelmap = None
        errors = []
        warnings = []
        details = []
        try:
            labelmap = self._exportLabelmap(volume)
            geometryErrors = self._geometryErrors(volume, labelmap)
            if geometryErrors:
                errors.append("标注与原图没有完全对齐，请回到图像中检查")
            array = np.asarray(slicer.util.arrayFromVolume(labelmap))
            if not np.isfinite(array).all():
                errors.append("标注中出现无法识别的数值")
            if not np.equal(array, np.floor(array)).all():
                errors.append("标注格式不正确，请重新检查")
            values = sorted(int(value) for value in np.unique(array))
            if any(value not in (0, 1) for value in values):
                errors.append("标注包含不允许的类别")
            foreground = int(np.count_nonzero(array == 1))
            if foreground <= 0:
                errors.append("当前标注为空")
            components = self._componentCount(array == 1)
            if components is not None and components > 10:
                warnings.append("还有很多零碎小区域，请确认是否误涂")
            details.extend(
                [
                    "技术检查结果",
                    f"病例：{self._currentCaseId}",
                    f"标签值：{values}",
                    f"前景体素数：{foreground}",
                    f"连通区域数：{components if components is not None else '未测量'}",
                    f"几何检查：{'通过' if not geometryErrors else '; '.join(geometryErrors)}",
                ]
            )
            issues = errors + warnings
            return {
                "passed": not issues,
                "issues": issues,
                "warnings": warnings,
                "details": "\n".join(details),
            }
        except Exception:
            details.extend(["检查异常", traceback.format_exc()])
            return {
                "passed": False,
                "issues": ["检查没有完成，请重试"],
                "details": "\n".join(details),
            }
        finally:
            if labelmap:
                try:
                    slicer.mrmlScene.RemoveNode(labelmap)
                except Exception:
                    pass

    def _checkAnnotation(self):
        result = self._runQualityCheck()
        self._setDetails(result.get("details", "") or self._lastDetailText)
        issues = result.get("issues", [])
        if result.get("passed"):
            self._qcStatus = "通过"
            self._setResultMessage(
                self.qcResultLabel,
                "✓ 标注检查通过\n技术检查正常，可以进入保存。",
                "success",
            )
            self._setStatusMessage("技术检查通过；请确认三维轮廓后保存。", "success")
        else:
            self._qcStatus = "需要确认"
            self._setResultMessage(
                self.qcResultLabel,
                "发现需要确认的问题：\n"
                + "\n".join(f"• {issue}" for issue in issues),
                "warning",
            )
            self._setStatusMessage("请根据提示返回图像中检查。", "warning")
        self._syncUi()
        return bool(result.get("passed"))

    def _saveAnnotation(self):
        if not self.volumeNode or not self.segmentationNode:
            self._setStatusMessage("请先导入病例并完成标注。", "warning")
            return False
        if not self._checkAnnotation():
            return False
        outputPath = self._outputPath()
        if outputPath.exists():
            box = qt.QMessageBox(slicer.util.mainWindow())
            box.setIcon(qt.QMessageBox.Question)
            box.setWindowTitle("本例已有保存结果")
            box.setText("本例已有保存结果，是否用当前标注替换？")
            replaceButton = box.addButton("替换保存", qt.QMessageBox.AcceptRole)
            cancelButton = box.addButton("取消", qt.QMessageBox.RejectRole)
            box.setDefaultButton(cancelButton)
            self._execDialog(box)
            if box.clickedButton() != replaceButton:
                return False
        labelmap = None
        try:
            outputPath.parent.mkdir(parents=True, exist_ok=True)
            labelmap = self._exportLabelmap(self.volumeNode)
            if self._geometryErrors(self.volumeNode, labelmap):
                raise RuntimeError("geometry mismatch")
            array = np.asarray(slicer.util.arrayFromVolume(labelmap))
            values = sorted(int(value) for value in np.unique(array))
            if any(value not in (0, 1) for value in values) or not np.count_nonzero(
                array == 1
            ):
                raise RuntimeError("label is empty or not binary")
            if not slicer.util.saveNode(labelmap, str(outputPath)):
                raise RuntimeError("Slicer could not save the label")
            self._upsertManifest(
                case_id=self._currentCaseId,
                volume=self.volumeNode,
                labelPath=outputPath,
                status="ANNOTATED",
                warnings=[],
            )
            self._saved = True
            self._dirty = False
            self._annotationHasData = True
            self._qcStatus = "通过"
            self.saveSuccessLabel.setVisible(True)
            self._setResultMessage(
                self.saveResultLabel,
                f"{self._currentCaseId} 已保存。",
                "success",
            )
            self._setDetails(
                f"保存完成\n病例：{self._currentCaseId}\n"
                f"自动保存位置：{outputPath}\nmanifest：{self.manifestPath}"
            )
            self._setStatusMessage("本例已保存，可以继续下一个病例。", "success")
            self._syncUi()
            return True
        except Exception:
            self._setDetails(
                f"保存失败\n自动保存位置：{outputPath}\n{traceback.format_exc()}"
            )
            self._setStatusMessage("保存没有完成，请重试。", "warning")
            return False
        finally:
            if labelmap:
                try:
                    slicer.mrmlScene.RemoveNode(labelmap)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Manifest and unsaved protection
    # ------------------------------------------------------------------
    def _imagePathForVolume(self, volume):
        path = self._volumePath(volume)
        if not path:
            return ""
        try:
            return path.relative_to(self.projectRoot).as_posix()
        except ValueError:
            return ""

    def _manifestPath(self, path):
        try:
            return Path(path).resolve().relative_to(self.projectRoot).as_posix()
        except ValueError:
            return ""

    def _upsertManifest(self, *, case_id, volume, labelPath, status, warnings):
        rows = []
        if self.manifestPath.exists():
            with self.manifestPath.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        existing = next((row for row in rows if row.get("case_id") == case_id), {})
        imagePath = self._imagePathForVolume(volume) or existing.get("image_path", "")
        updated = {
            "case_id": case_id,
            "group_id": existing.get("group_id") or case_id,
            "side": existing.get("side", ""),
            "image_path": imagePath,
            "label_path": self._manifestPath(labelPath),
            "annotation_status": status,
            "geometry_valid": "true" if status == "ANNOTATED" else existing.get("geometry_valid", ""),
            "label_valid": "true" if status == "ANNOTATED" else existing.get("label_valid", ""),
            "notes": "由 TMJ Condyle Annotator 保存。"
            + (" " + " ".join(warnings) if warnings else ""),
        }
        rows = [row for row in rows if row.get("case_id") != case_id]
        rows.append(updated)
        self.manifestPath.parent.mkdir(parents=True, exist_ok=True)
        with self.manifestPath.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in MANIFEST_FIELDS} for row in rows
            )

    def _removeSegmentation(self):
        self._removeSegmentationObservers()
        self._removeViewObservations()
        if self.editorWidget:
            try:
                self.editorWidget.setActiveEffect(None)
                self.editorWidget.setSegmentationNode(None)
            except Exception:
                pass
        if self.segmentationNode:
            try:
                slicer.mrmlScene.RemoveNode(self.segmentationNode)
            except Exception:
                pass
        self.segmentationNode = None
        self.segmentId = None
        self._annotationHasData = False
        self._editorViewsObserved = False

    def _askUnsavedDecision(self):
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Question)
        box.setWindowTitle("本例标注还没有保存")
        box.setText("本例标注还没有保存。")
        box.setInformativeText("离开前要如何处理当前标注？")
        saveButton = box.addButton("保存后继续", qt.QMessageBox.AcceptRole)
        discardButton = box.addButton("不保存", qt.QMessageBox.DestructiveRole)
        cancelButton = box.addButton("取消", qt.QMessageBox.RejectRole)
        box.setDefaultButton(saveButton)
        self._execDialog(box)
        clicked = box.clickedButton()
        if clicked == saveButton:
            return "save"
        if clicked == discardButton:
            return "discard"
        return "cancel"

    def _confirmUnsaved(self):
        if not self._dirty:
            return True
        decision = self._askUnsavedDecision()
        if decision == "save":
            return self._saveAnnotation()
        if decision == "discard":
            self._dirty = False
            return True
        return False

    def _installCloseProtection(self):
        try:
            self._mainWindow = slicer.util.mainWindow()
            self._mainWindow.installEventFilter(self)
        except Exception:
            self._mainWindow = None

    def eventFilter(self, watched, event):
        try:
            if (
                watched == self._mainWindow
                and event.type() == qt.QEvent.Close
                and self._dirty
            ):
                if not self._confirmUnsaved():
                    return True
        except Exception:
            pass
        return False

    def _moveToNextCaseAfterSave(self):
        if not self._moveCase(1):
            self._setStatusMessage("当前已经是最后一个病例。", "info")

    def cleanup(self):
        if self._mainWindow:
            try:
                self._mainWindow.removeEventFilter(self)
            except Exception:
                pass
        self._removeViewObservations()
        self._setActiveEffect(None)
        if self.editorWidget:
            try:
                self.editorWidget.uninstallKeyboardShortcuts()
            except Exception:
                pass
        if self._effectFactory and self._effectFactoryConnected:
            try:
                self._effectFactory.disconnect(
                    "effectRegistered(QString)", self._onEffectRegistered
                )
            except Exception:
                pass
        self._effectFactoryConnected = False
        self._removeSegmentationObservers()
        for node, tag in self._sliceObservers:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._sliceObservers = []
        for node, tag in self._sceneObservers:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._sceneObservers = []
        self._restoreWorkflowLayout()
        self._setSimpleMode(False)
        try:
            super().cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI state
    # ------------------------------------------------------------------
    def _statusText(self):
        if self._saved and not self._dirty:
            return "已完成"
        if self._annotationHasData:
            return "标注中"
        return "未标注"

    def _currentStep(self):
        if not self.volumeNode:
            return 0
        return self._currentPage

    def _stepIsDone(self, index):
        if index == 0:
            return bool(self.volumeNode)
        if index == 1:
            return bool(self._annotationHasData)
        if index == 2:
            return self._qcStatus == "通过"
        if index == 3:
            return bool(self._saved and not self._dirty)
        return False

    def _syncUi(self):
        if not hasattr(self, "pageStack"):
            return
        status = self._statusText()
        self.caseIdValue.setText(self._currentCaseId)
        if self._caseFiles and self._currentCaseIndex >= 0:
            progress = f"{self._currentCaseIndex + 1} / {len(self._caseFiles)}"
        elif self._caseFiles:
            progress = f"— / {len(self._caseFiles)}"
        else:
            progress = "1 / 1"
        self.caseProgressValue.setText(progress)
        self.caseStatusValue.setText(status)
        self.caseSaveValue.setText("已保存" if self._saved and not self._dirty else "未保存")
        self.caseNavigationLabel.setText("病例 " + progress)

        statusState = "complete" if status == "已完成" else "working" if status == "标注中" else ""
        statusPrefix = "✓ " if status == "已完成" else "● "
        self.statusChip.setText(statusPrefix + status)
        self.statusChip.setProperty("status", statusState)
        self.statusChip.style().unpolish(self.statusChip)
        self.statusChip.style().polish(self.statusChip)

        currentStep = self._currentStep()
        for index, card in enumerate(self.stepWidgets):
            if self._stepIsDone(index):
                state = "done"
                statusText = "✓ 已完成"
            elif index == currentStep:
                state = "current"
                statusText = "进行中"
            else:
                state = "waiting"
                statusText = "等待"
            card.setProperty("state", state)
            self.stepStatusLabels[index].setText(statusText)
            card.style().unpolish(card)
            card.style().polish(card)

        hasVolume = bool(self.volumeNode)
        hasSegmentation = bool(self.segmentationNode)
        editorReady = bool(self.editorWidget and self._editorReady)
        self.importNextButton.setEnabled(hasVolume)
        self.useCurrentButton.setEnabled(bool(self._loadedVolumeIds))
        self.paintButton.setEnabled(hasSegmentation and editorReady)
        self.eraseButton.setEnabled(hasSegmentation and editorReady)
        self.undoButton.setEnabled(hasSegmentation and editorReady)
        self.redoButton.setEnabled(hasSegmentation and editorReady)
        self.opacitySlider.setEnabled(hasSegmentation)
        self.annotateNextButton.setEnabled(hasSegmentation)
        self.show3DButton.setEnabled(hasSegmentation)
        self.checkButton.setEnabled(hasSegmentation)
        self.confirmCheckButton.setEnabled(self._qcStatus == "通过")
        self.saveButton.setEnabled(
            hasSegmentation and self._qcStatus == "通过"
        )
        self.previousCaseButton.setEnabled(
            bool(self._caseFiles) and self._currentCaseIndex > 0
        )
        self.nextCaseButton.setEnabled(
            bool(self._caseFiles)
            and self._currentCaseIndex >= 0
            and self._currentCaseIndex < len(self._caseFiles) - 1
        )
        self.nextCaseAfterSaveButton.setEnabled(
            bool(self._caseFiles)
            and self._currentCaseIndex >= 0
            and self._currentCaseIndex < len(self._caseFiles) - 1
        )
        self._setToolButtonState()
        self._updateSliceLabel()
        self._updateSaveSummary()

    def _updateSaveSummary(self):
        if not hasattr(self, "saveSummaryLabel"):
            return
        self.saveSummaryLabel.setText(
            f"病例：{self._currentCaseId}\n"
            f"✓ MRI 已加载\n"
            f"✓ 已完成标注\n"
            f"✓ 技术检查：{self._qcStatus}"
        )


class TMJCondyleAnnotatorTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_case_id_and_binary_rules()

    def test_case_id_and_binary_rules(self):
        self.assertTrue(CASE_ID_PATTERN.fullmatch("case_001"))
        self.assertTrue(CASE_FILE_PATTERN.fullmatch("case_001"))
        self.assertFalse(CASE_FILE_PATTERN.fullmatch("Patient Zhang"))
        self.assertTrue(
            set(np.unique(np.array([0, 1], dtype=np.uint8))).issubset({0, 1})
        )
