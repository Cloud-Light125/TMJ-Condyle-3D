"""Chinese four-step workbench for manual mandibular-condyle annotation.

The clinical-facing workflow lives in this module.  A qMRMLSegmentEditorWidget
is kept as an embedded editing engine, but the native Segment Editor module is
never selected or shown to the user.  The module owns the Segment Editor
parameter node, the MRML scene association, and slice-view observations.
"""

from __future__ import annotations

import csv
import datetime
import json
import re
import shutil
import sys
import time
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


# The module is commonly loaded through Slicer's additional-module-path.  Add
# the project root explicitly so the GUI and the command-line workflow share
# the same tested experiment service layer in both launch modes.
_PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

from tmj_condyle.experiment import (  # noqa: E402
    FOLDS,
    assess_training_readiness,
    case_counts,
    completed_folds,
    count_guidance,
    create_experiment_run,
    dataset_build_command,
    dataset_validation_command,
    detect_fold_states,
    environment_command,
    environment_display,
    evaluation_command,
    export_experiment_results,
    finalize_experiment_run,
    fold_results_directory,
    format_metric,
    has_evaluation_results,
    home_next_step,
    home_next_action,
    list_experiment_runs,
    load_case_inventory,
    oof_command,
    parse_environment_json,
    parse_training_line,
    prediction_command,
    prediction_result_ready,
    project_python_executable,
    read_experiment_record,
    read_metrics_csv,
    read_metrics_summary,
    read_splits,
    read_validation_csv,
    script_command,
    summarize_metrics_by_fold,
    training_command,
    training_prerequisite_summary,
    user_training_message,
    should_show_first_run_wizard,
)
from tmj_condyle.data.manifest import read_manifest  # noqa: E402
from tmj_condyle.launcher import (  # noqa: E402
    configured_slicer_path,
    discover_slicer_candidates,
    slicer_config_path,
    write_slicer_config,
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

MAIN_NAVIGATION = (
    ("home", "首页"),
    ("cases", "病例与标注"),
    ("dataset", "训练数据"),
    ("training", "模型训练"),
    ("results", "实验结果"),
    ("prediction", "自动分割"),
    ("settings", "设置"),
)
PAGE_HOME = 0
PAGE_CASE_IMPORT = 1
PAGE_CASE_ANNOTATION = 2
PAGE_CASE_CHECK = 3
PAGE_CASE_SAVE = 4
PAGE_DATASET = 5
PAGE_TRAINING = 6
PAGE_RESULTS = 7
PAGE_PREDICTION = 8
PAGE_SETTINGS = 9


class TMJCondyleAnnotator(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "下颌髁突三维分割实验平台"
        self.parent.categories = ["Segmentation"]
        self.parent.contributors = ["TMJ-Condyle-3D"]
        self.parent.helpText = (
            "面向牙科医学生的下颌髁突 MRI 三维分割实验平台。"
            "从病例导入、人工标注到 nnU-Net 训练、评价和新病例自动分割，"
            "所有步骤都在中文工作台内完成。"
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
        self._currentAnnotationStatus = "NEW"
        self._currentEffectName = "Paint"
        self._currentCaseId = "case_001"
        self._currentCasePath = None
        self._currentCaseIndex = 0
        self._caseFiles = []
        self._batchCaseFiles = None
        self._batchCaseIds = {}
        self._lastDetailText = ""
        self._currentPage = 0
        self._homeVisible = True
        self._primaryAxis = 2
        self._primaryOrientation = "Axial"
        self._threeDVisible = False
        self._surfacePointCount = 0
        self._surfaceCellCount = 0

        # Experiment-platform state.  The annotation editor state above is
        # intentionally kept separate so the mature editor remains stable.
        self._mainPage = "home"
        self._process = None
        self._processKind = ""
        self._processOutput = ""
        self._processLogPath = None
        self._processStartedAt = None
        self._processStopRequested = False
        self._environmentReport = None
        self._validationRows = read_validation_csv(
            self.projectRoot / "workspace" / "reports" / "dataset_validation.csv"
        )
        self._validationPassed = False
        self._datasetPrepared = False
        self._preprocessingPrepared = False
        self._trainingFoldEvents = {}
        self._activeTrainingFold = None
        self._currentRunDir = None
        self._resultRunDir = None
        self._resultRows = []
        self._resultSummary = {}
        self._selectedResultCase = None
        self._predictionInputPath = None
        self._predictionOutputPath = None
        self._predictionImageNode = None
        self._predictionLabelNode = None
        self._predictionSegmentationNode = None
        self._resultImageNode = None
        self._resultGroundTruthNode = None
        self._resultPredictionNode = None
        self._resultCompareSegmentationNode = None
        self._taskTimer = None
        self._firstRunWizardDialog = None
        self._slicerCandidates = []

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
        self._setStatusMessage("欢迎使用。先导入病例并标注下颌髁突。", "info")
        self._syncUi()
        # Show the novice orientation only after the module widget exists and
        # Slicer has had a chance to finish constructing its main window.
        qt.QTimer.singleShot(700, self._showFirstRunWizardIfNeeded)

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
        QFrame#summaryCard, QLabel#summaryCard, QFrame#navCard,
        QFrame#statCard, QFrame#foldCard, QFrame#metricCard {
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
        QLabel#homeLead {
          color: #163b57;
          font-size: 16px;
          font-weight: 600;
        }
        QLabel#homePurpose {
          color: #526777;
          font-size: 13px;
          line-height: 1.5;
        }
        QLabel#homeNextStep {
          color: #16839a;
          background: #eaf6f8;
          border-radius: 10px;
          padding: 10px 12px;
          font-size: 13px;
          font-weight: 600;
        }
        QLabel#statCaption {
          color: #718391;
          font-size: 11px;
        }
        QLabel#statValue {
          color: #163b57;
          font-size: 19px;
          font-weight: 600;
        }
        QLabel#metricValue {
          color: #163b57;
          font-size: 20px;
          font-weight: 600;
        }
        QLabel#metricCaption {
          color: #718391;
          font-size: 11px;
        }
        QPushButton#navButton {
          background: transparent;
          color: #526777;
          border: none;
          border-radius: 8px;
          min-height: 34px;
          padding: 0 9px;
          font-size: 12px;
          font-weight: 600;
        }
        QPushButton#navButton:hover {
          background: #eef6fb;
          color: #16839a;
        }
        QPushButton#navButton[active="true"] {
          background: #16839a;
          color: #ffffff;
        }
        QLabel#statusMessage, QLabel#resultMessage, QLabel#infoMessage {
          min-height: 18px;
        }
        QLabel#homeProgress {
          color: #28546a;
          font-size: 13px;
          font-weight: 600;
          padding: 8px 0;
        }
        QPushButton#homeCardButton {
          background: #ffffff;
          color: #28546a;
          border: 1px solid #c8dbe2;
          border-radius: 12px;
          min-height: 58px;
          padding: 0 14px;
          font-size: 14px;
          font-weight: 600;
          text-align: left;
        }
        QPushButton#homeCardButton:hover {
          background: #f2f8fa;
          border-color: #8dbbc7;
        }
        QPushButton#homeCardButton:disabled {
          color: #a6b4bc;
          border-color: #e1e8ec;
          background: #fafbfc;
        }
        QLabel#viewHint {
          color: #526777;
          background: #eef6fb;
          border-radius: 10px;
          padding: 10px 12px;
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
        QPlainTextEdit#taskLog {
          background: #17232d;
          color: #d5e6ec;
          border: none;
          border-radius: 9px;
          font-size: 11px;
        }
        QProgressBar#taskProgress {
          background: #eef2f5;
          border: none;
          border-radius: 5px;
          height: 9px;
          text-align: center;
        }
        QProgressBar#taskProgress::chunk {
          background: #16839a;
          border-radius: 5px;
        }
        QTableWidget, QListWidget {
          background: #ffffff;
          border: 1px solid #e1eaf0;
          border-radius: 9px;
          alternate-background-color: #f7fafb;
        }
        QTableWidget::item:selected, QListWidget::item:selected {
          background: #dff1f3;
          color: #163b57;
        }
        QHeaderView::section {
          background: #eef6fb;
          color: #526777;
          border: none;
          padding: 6px;
          font-weight: 600;
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
        self._buildMainNavigation(mainLayout)
        self._buildStepBar(mainLayout)
        self._buildCaseCard(mainLayout)

        self.pageStack = qt.QStackedWidget()
        self.pageStack.setObjectName("workflowPages")
        self.pageStack.addWidget(self._buildHomePage())
        self.pageStack.addWidget(self._buildImportPage())
        self.pageStack.addWidget(self._buildAnnotationPage())
        self.pageStack.addWidget(self._buildCheckPage())
        self.pageStack.addWidget(self._buildSavePage())
        self.pageStack.addWidget(self._buildDatasetPage())
        self.pageStack.addWidget(self._buildTrainingPage())
        self.pageStack.addWidget(self._buildResultsPage())
        self.pageStack.addWidget(self._buildPredictionPage())
        self.pageStack.addWidget(self._buildSettingsPage())
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

    def _buildMainNavigation(self, mainLayout):
        card = self._card("navCard")
        row = qt.QHBoxLayout(card)
        row.setContentsMargins(7, 5, 7, 5)
        row.setSpacing(2)
        self.navigationButtons = {}
        for key, label in MAIN_NAVIGATION:
            button = qt.QPushButton(label)
            button.setObjectName("navButton")
            button.setProperty("active", key == "home")
            button.clicked.connect(
                lambda checked=False, page=key: self._showMainPage(page)
            )
            self.navigationButtons[key] = button
            row.addWidget(button, 1)
        self.navigationCard = card
        mainLayout.addWidget(card)

    def _buildHeader(self, mainLayout):
        card = self._card("headerCard")
        row = qt.QHBoxLayout(card)
        row.setContentsMargins(18, 14, 14, 14)
        brand = qt.QVBoxLayout()
        self.titleLabel = qt.QLabel("下颌髁突三维分割实验平台")
        self.titleLabel.setObjectName("mainTitle")
        brand.addWidget(self.titleLabel)
        self.subtitleLabel = qt.QLabel("TMJ MRI · 标注、训练、评估与三维分割")
        self.subtitleLabel.setObjectName("subtitleLabel")
        brand.addWidget(self.subtitleLabel)
        row.addLayout(brand, 1)

        right = qt.QVBoxLayout()
        actionRow = qt.QHBoxLayout()
        self.homeButton = self._linkButton("返回首页")
        self.homeButton.clicked.connect(self._goHome)
        actionRow.addWidget(self.homeButton)
        self.helpButton = self._linkButton("？ 使用帮助")
        self.helpButton.clicked.connect(self._showUsageGuide)
        actionRow.addWidget(self.helpButton)
        self.simpleModeButton = self._linkButton("显示完整 Slicer")
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
        container = qt.QWidget()
        row = qt.QHBoxLayout(container)
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
        self.stepBarWidget = container
        mainLayout.addWidget(container)

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
        self.caseSummaryCard = card
        mainLayout.addWidget(card)

    def _buildHomePage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        welcome = qt.QLabel("欢迎使用")
        welcome.setObjectName("pageEyebrow")
        layout.addWidget(welcome)
        title = qt.QLabel("下颌髁突三维分割实验平台")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        subtitle = qt.QLabel(
            "TMJ MRI · 标注、训练、评估与三维分割"
        )
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        lead = qt.QLabel("从人工标注到自动分割，一站完成实验：")
        lead.setObjectName("homeLead")
        layout.addWidget(lead)
        purpose = qt.QLabel(
            "先导入核磁并标出下颌髁突，再准备训练数据。\n"
            "系统会调用真实的 nnU-Net v2 3d_fullres 和 grouped 5-fold 流程。\n"
            "训练完成后，可以查看 Dice / IoU / HD95，并用新核磁自动分割。"
        )
        purpose.setObjectName("homePurpose")
        purpose.setWordWrap(True)
        layout.addWidget(purpose)

        self.homeProgressLabel = qt.QLabel()
        self.homeProgressLabel.setObjectName("homeProgress")
        self.homeProgressLabel.setWordWrap(True)
        layout.addWidget(self.homeProgressLabel)

        self.homeNextStepLabel = qt.QLabel()
        self.homeNextStepLabel.setObjectName("homeNextStep")
        self.homeNextStepLabel.setWordWrap(True)
        layout.addWidget(self.homeNextStepLabel)
        self.homeNextActionButton = self._primaryButton("开始导入病例")
        self.homeNextActionButton.clicked.connect(self._runHomeNextAction)
        layout.addWidget(self.homeNextActionButton, 0, qt.Qt.AlignLeft)

        statsCard = self._card("statCard")
        statsLayout = qt.QHBoxLayout(statsCard)
        statsLayout.setContentsMargins(12, 10, 12, 10)
        self.homeStatLabels = {}
        for key, caption in (
            ("total", "病例总数"),
            ("manual", "人工标注完成"),
            ("verified", "已验证标注"),
            ("trainable", "可用于训练"),
            ("training", "训练状态"),
            ("results", "实验结果"),
            ("model", "训练模型"),
        ):
            column = qt.QVBoxLayout()
            captionLabel = qt.QLabel(caption)
            captionLabel.setObjectName("statCaption")
            valueLabel = qt.QLabel("—")
            valueLabel.setObjectName("statValue")
            column.addWidget(captionLabel)
            column.addWidget(valueLabel)
            statsLayout.addLayout(column, 1)
            self.homeStatLabels[key] = valueLabel
        layout.addWidget(statsCard)

        workflowCard = self._card("hintCard")
        workflowLayout = qt.QVBoxLayout(workflowCard)
        workflowLayout.setContentsMargins(14, 12, 14, 12)
        workflowTitle = qt.QLabel("实验流程状态")
        workflowTitle.setObjectName("sectionTitle")
        workflowLayout.addWidget(workflowTitle)
        self.homeWorkflowLabel = qt.QLabel()
        self.homeWorkflowLabel.setObjectName("hintLabel")
        self.homeWorkflowLabel.setWordWrap(True)
        workflowLayout.addWidget(self.homeWorkflowLabel)
        layout.addWidget(workflowCard)

        cards = qt.QGridLayout()
        cards.setHorizontalSpacing(10)
        cards.setVerticalSpacing(10)
        self.homeStartButton = self._homeCardButton("继续标注")
        self.homeStartButton.clicked.connect(self._continueLastAnnotation)
        cards.addWidget(self.homeStartButton, 0, 0)
        self.homeContinueButton = self._homeCardButton("导入病例")
        self.homeContinueButton.clicked.connect(self._startNewAnnotationFromHome)
        cards.addWidget(self.homeContinueButton, 0, 1)
        self.homeProgressButton = self._homeCardButton("病例与标注")
        self.homeProgressButton.clicked.connect(lambda checked=False: self._showMainPage("cases"))
        cards.addWidget(self.homeProgressButton, 1, 0)
        self.homeAnnotatedButton = self._homeCardButton("准备训练")
        self.homeAnnotatedButton.clicked.connect(lambda checked=False: self._showMainPage("dataset"))
        cards.addWidget(self.homeAnnotatedButton, 1, 1)
        self.homeTrainingButton = self._homeCardButton("开始实验")
        self.homeTrainingButton.clicked.connect(lambda checked=False: self._showMainPage("training"))
        cards.addWidget(self.homeTrainingButton, 2, 0)
        self.homeResultsButton = self._homeCardButton("查看实验结果")
        self.homeResultsButton.clicked.connect(lambda checked=False: self._showMainPage("results"))
        cards.addWidget(self.homeResultsButton, 2, 1)
        self.homePredictButton = self._homeCardButton("自动分割新病例")
        self.homePredictButton.clicked.connect(lambda checked=False: self._showMainPage("prediction"))
        cards.addWidget(self.homePredictButton, 3, 0)
        self.homeHelpButton = self._homeCardButton("怎么做实验？")
        self.homeHelpButton.clicked.connect(self._showUsageGuide)
        cards.addWidget(self.homeHelpButton, 3, 1)
        self.homeDemoButton = self._homeCardButton("查看软件演示")
        self.homeDemoButton.clicked.connect(self._showDemoDialog)
        cards.addWidget(self.homeDemoButton, 4, 0)
        self.homeSettingsButton = self._homeCardButton("平台设置")
        self.homeSettingsButton.clicked.connect(lambda checked=False: self._showMainPage("settings"))
        cards.addWidget(self.homeSettingsButton, 4, 1)
        layout.addLayout(cards)

        processCard = self._card("hintCard")
        processLayout = qt.QVBoxLayout(processCard)
        processLayout.setContentsMargins(14, 12, 14, 12)
        processTitle = qt.QLabel("完整实验流程")
        processTitle.setObjectName("sectionTitle")
        processLayout.addWidget(processTitle)
        process = qt.QLabel("① 准备病例　→　② 标注髁突　→　③ 训练模型　→　④ 查看结果　→　⑤ 自动分割新病例")
        process.setObjectName("hintLabel")
        process.setWordWrap(True)
        processLayout.addWidget(process)
        layout.addWidget(processCard)
        layout.addStretch(1)
        return page

    @staticmethod
    def _homeCardButton(text):
        button = qt.QPushButton(text)
        button.setObjectName("homeCardButton")
        return button

    def _buildImportPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "第 1 步",
            "选择病例",
            "选择一份需要标注的核磁，或一次导入一个病例文件夹。",
        )

        self.importLoadedRow = qt.QHBoxLayout()
        loadedCaption = qt.QLabel("当前已加载")
        loadedCaption.setObjectName("mutedLabel")
        self.importLoadedRow.addWidget(loadedCaption)
        self.loadedVolumeSelector = qt.QComboBox()
        self.loadedVolumeSelector.currentIndexChanged.connect(
            self._onLoadedVolumeSelected
        )
        self.importLoadedRow.addWidget(self.loadedVolumeSelector, 1)
        layout.addLayout(self.importLoadedRow)

        caseTitle = qt.QLabel("病例列表")
        caseTitle.setObjectName("sectionTitle")
        layout.addWidget(caseTitle)
        self.caseListWidget = qt.QListWidget()
        self.caseListWidget.setAlternatingRowColors(True)
        self.caseListWidget.setMinimumHeight(120)
        self.caseListWidget.currentRowChanged.connect(self._onCaseListRowChanged)
        layout.addWidget(self.caseListWidget)

        caseActionRow = qt.QHBoxLayout()
        self.caseContinueButton = self._primaryButton("继续标注")
        self.caseContinueButton.clicked.connect(self._continueSelectedCase)
        caseActionRow.addWidget(self.caseContinueButton)
        self.caseViewButton = self._secondaryButton("查看标注")
        self.caseViewButton.clicked.connect(self._viewSelectedCase)
        caseActionRow.addWidget(self.caseViewButton)
        self.caseReeditButton = self._secondaryButton("重新编辑")
        self.caseReeditButton.clicked.connect(self._reeditSelectedCase)
        caseActionRow.addWidget(self.caseReeditButton)
        layout.addLayout(caseActionRow)

        buttonRow = qt.QHBoxLayout()
        self.loadButton = self._primaryButton("选择核磁文件")
        self.loadButton.clicked.connect(self._chooseAndLoadVolume)
        buttonRow.addWidget(self.loadButton)
        self.folderButton = self._secondaryButton("导入病例文件夹")
        self.folderButton.clicked.connect(self._chooseAndLoadFolder)
        buttonRow.addWidget(self.folderButton)
        self.useCurrentButton = self._secondaryButton("继续当前病例")
        self.useCurrentButton.clicked.connect(self._useSelectedLoadedVolume)
        buttonRow.addWidget(self.useCurrentButton)
        layout.addLayout(buttonRow)

        self.importResultLabel = qt.QLabel()
        self.importResultLabel.setObjectName("resultMessage")
        self.importResultLabel.setWordWrap(True)
        layout.addWidget(self.importResultLabel)

        self.importNextButton = self._primaryButton("下一步：标注髁突")
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
            "标注髁突",
            "用画笔把图像里的下颌髁突涂出来。建议从第一层检查到最后一层。",
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
        self.currentToolLabel = qt.QLabel("正在使用：画笔")
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

        self.sliceSuggestionLabel = qt.QLabel(
            "建议从第一层检查到最后一层，确保每一层都没有漏标。"
        )
        self.sliceSuggestionLabel.setObjectName("hintLabel")
        self.sliceSuggestionLabel.setWordWrap(True)
        layout.addWidget(self.sliceSuggestionLabel)

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

        self.annotateNextButton = self._primaryButton("标完了，检查三维")
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
            "检查三维结果",
            "这是你刚才标出的下颌髁突三维形状。转动查看有没有缺一块、多出小块，整体是否连续。",
        )

        checkCard = self._card("hintCard")
        checkLayout = qt.QVBoxLayout(checkCard)
        checkLayout.setContentsMargins(13, 11, 13, 11)
        self.checkListLabel = qt.QLabel(
            "请检查：\n"
            "• 有没有明显缺一块\n"
            "• 有没有多出奇怪的小块\n"
            "• 整体形状是否连续"
        )
        self.checkListLabel.setObjectName("hintLabel")
        self.checkListLabel.setWordWrap(True)
        checkLayout.addWidget(self.checkListLabel)
        buttonRow = qt.QHBoxLayout()
        self.show3DButton = self._primaryButton("检查三维效果")
        self.show3DButton.clicked.connect(self._show3D)
        buttonRow.addWidget(self.show3DButton)
        self.redisplay3DButton = self._secondaryButton("重新显示 3D")
        self.redisplay3DButton.clicked.connect(self._show3D)
        buttonRow.addWidget(self.redisplay3DButton)
        self.checkButton = self._secondaryButton("检查标注")
        self.checkButton.clicked.connect(self._checkAnnotation)
        buttonRow.addWidget(self.checkButton)
        checkLayout.addLayout(buttonRow)
        layout.addWidget(checkCard)

        viewSwitchRow = qt.QHBoxLayout()
        viewLabel = qt.QLabel("查看方式")
        viewLabel.setObjectName("sectionTitle")
        viewSwitchRow.addWidget(viewLabel)
        self.view2DButton = self._secondaryButton("看切片")
        self.view2DButton.clicked.connect(self._show2DCheckView)
        viewSwitchRow.addWidget(self.view2DButton)
        self.view3DButton = self._secondaryButton("看三维")
        self.view3DButton.clicked.connect(self._show3D)
        viewSwitchRow.addWidget(self.view3DButton)
        viewSwitchRow.addStretch(1)
        layout.addLayout(viewSwitchRow)

        self.threeDHintLabel = qt.QLabel(
            "鼠标左键拖动：旋转　　鼠标滚轮：放大 / 缩小"
        )
        self.threeDHintLabel.setObjectName("viewHint")
        self.threeDHintLabel.setWordWrap(True)
        layout.addWidget(self.threeDHintLabel)

        self.qcResultLabel = qt.QLabel()
        self.qcResultLabel.setObjectName("resultMessage")
        self.qcResultLabel.setWordWrap(True)
        layout.addWidget(self.qcResultLabel)

        checkNavigation = qt.QHBoxLayout()
        self.backToAnnotationButton = self._secondaryButton("返回修改标注")
        self.backToAnnotationButton.clicked.connect(self._returnToAnnotation)
        checkNavigation.addWidget(self.backToAnnotationButton)
        checkNavigation.addStretch(1)
        self.confirmCheckButton = self._primaryButton("没问题，下一步保存")
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
            "确认检查完成后，保存本例的标注结果。",
        )

        self.saveSummaryLabel = qt.QLabel()
        self.saveSummaryLabel.setObjectName("summaryCard")
        self.saveSummaryLabel.setWordWrap(True)
        self.saveSummaryLabel.setMargin(14)
        layout.addWidget(self.saveSummaryLabel)

        self.saveButton = self._primaryButton("保存本例")
        self.saveButton.clicked.connect(self._saveAnnotation)
        layout.addWidget(self.saveButton)

        self.confirmAnnotationButton = self._primaryButton("确认本例标注")
        self.confirmAnnotationButton.clicked.connect(self._confirmAnnotation)
        layout.addWidget(self.confirmAnnotationButton)
        self.confirmAnnotationHintLabel = qt.QLabel(
            "确认后，这一例可以用于训练。确认前请由牙医或项目负责人完成复核。"
        )
        self.confirmAnnotationHintLabel.setObjectName("mutedLabel")
        self.confirmAnnotationHintLabel.setWordWrap(True)
        layout.addWidget(self.confirmAnnotationHintLabel)

        self.saveResultLabel = qt.QLabel()
        self.saveResultLabel.setObjectName("resultMessage")
        self.saveResultLabel.setWordWrap(True)
        layout.addWidget(self.saveResultLabel)
        self.saveSuccessLabel = qt.QLabel()
        self.saveSuccessLabel.setObjectName("bigSuccess")
        self.saveSuccessLabel.setAlignment(qt.Qt.AlignCenter)
        self.saveSuccessLabel.setVisible(False)
        layout.addWidget(self.saveSuccessLabel)

        self.saveNextGuidanceLabel = qt.QLabel()
        self.saveNextGuidanceLabel.setObjectName("resultMessage")
        self.saveNextGuidanceLabel.setWordWrap(True)
        layout.addWidget(self.saveNextGuidanceLabel)

        saveNavigation = qt.QHBoxLayout()
        self.backToCheckButton = self._secondaryButton("返回检查")
        self.backToCheckButton.clicked.connect(self._returnToCheck)
        saveNavigation.addWidget(self.backToCheckButton)
        saveNavigation.addStretch(1)
        self.nextCaseAfterSaveButton = self._primaryButton("继续下一个病例")
        self.nextCaseAfterSaveButton.clicked.connect(self._moveToNextCaseAfterSave)
        saveNavigation.addWidget(self.nextCaseAfterSaveButton)
        layout.addLayout(saveNavigation)
        layout.addStretch(1)
        return page

    def _buildDatasetPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "实验准备",
            "训练数据",
            "先检查全部病例，再把通过检查的真实人工标注整理成训练数据。技术文件会由系统自动生成。",
        )

        self.datasetSummaryLabel = qt.QLabel()
        self.datasetSummaryLabel.setObjectName("resultMessage")
        self.datasetSummaryLabel.setWordWrap(True)
        layout.addWidget(self.datasetSummaryLabel)
        self.datasetGuidanceLabel = qt.QLabel()
        self.datasetGuidanceLabel.setObjectName("infoMessage")
        self.datasetGuidanceLabel.setWordWrap(True)
        layout.addWidget(self.datasetGuidanceLabel)
        self.datasetExcludedLabel = qt.QLabel()
        self.datasetExcludedLabel.setObjectName("mutedLabel")
        self.datasetExcludedLabel.setWordWrap(True)
        layout.addWidget(self.datasetExcludedLabel)

        stepCard = self._card("hintCard")
        stepLayout = qt.QVBoxLayout(stepCard)
        stepLayout.setContentsMargins(13, 11, 13, 11)
        stepTitle = qt.QLabel("准备进度")
        stepTitle.setObjectName("sectionTitle")
        stepLayout.addWidget(stepTitle)
        self.datasetStepLabels = []
        for textValue in ("检查病例", "整理训练数据", "生成 5 组实验"):
            label = qt.QLabel("○ " + textValue)
            label.setObjectName("mutedLabel")
            stepLayout.addWidget(label)
            self.datasetStepLabels.append(label)
        layout.addWidget(stepCard)

        self.datasetCaseTable = qt.QTableWidget(0, 4)
        self.datasetCaseTable.setHorizontalHeaderLabels(["病例", "状态", "患者组", "问题"])
        self.datasetCaseTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.datasetCaseTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.datasetCaseTable.setMaximumHeight(180)
        layout.addWidget(self.datasetCaseTable)

        buttonRow = qt.QHBoxLayout()
        self.checkDatasetButton = self._primaryButton("检查全部病例")
        self.checkDatasetButton.clicked.connect(self._startDatasetValidation)
        buttonRow.addWidget(self.checkDatasetButton)
        self.buildDatasetButton = self._secondaryButton("准备训练数据")
        self.buildDatasetButton.clicked.connect(self._startDatasetBuild)
        buttonRow.addWidget(self.buildDatasetButton)
        buttonRow.addStretch(1)
        layout.addLayout(buttonRow)

        self.datasetResultLabel = qt.QLabel()
        self.datasetResultLabel.setObjectName("resultMessage")
        self.datasetResultLabel.setWordWrap(True)
        layout.addWidget(self.datasetResultLabel)
        self.datasetAdvancedLabel = qt.QLabel()
        self.datasetAdvancedLabel.setObjectName("mutedLabel")
        self.datasetAdvancedLabel.setWordWrap(True)
        self.datasetAdvancedLabel.setVisible(False)
        layout.addWidget(self.datasetAdvancedLabel)
        layout.addStretch(1)
        return page

    def _buildTrainingPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "实验步骤",
            "模型训练",
            "默认使用 nnU-Net v2 的 3D 下颌髁突分割模型和 5 组交叉验证。普通模式不需要修改机器学习参数。",
        )

        modelCard = self._card("hintCard")
        modelLayout = qt.QFormLayout(modelCard)
        modelLayout.setContentsMargins(13, 11, 13, 11)
        modelLayout.addRow("模型：", qt.QLabel("3D 下颌髁突分割模型"))
        modelLayout.addRow("训练方法：", qt.QLabel("nnU-Net 3D"))
        modelLayout.addRow("实验方式：", qt.QLabel("5 组交叉验证"))
        layout.addWidget(modelCard)

        self.trainingEnvironmentLabel = qt.QLabel("正在准备系统检查…")
        self.trainingEnvironmentLabel.setObjectName("resultMessage")
        self.trainingEnvironmentLabel.setWordWrap(True)
        layout.addWidget(self.trainingEnvironmentLabel)
        self.trainingReadinessLabel = qt.QLabel()
        self.trainingReadinessLabel.setObjectName("infoMessage")
        self.trainingReadinessLabel.setWordWrap(True)
        layout.addWidget(self.trainingReadinessLabel)
        self.trainingGpuWarningLabel = qt.QLabel()
        self.trainingGpuWarningLabel.setObjectName("resultMessage")
        self.trainingGpuWarningLabel.setWordWrap(True)
        layout.addWidget(self.trainingGpuWarningLabel)

        foldTitle = qt.QLabel("训练进度")
        foldTitle.setObjectName("sectionTitle")
        layout.addWidget(foldTitle)
        self.trainingFoldList = qt.QListWidget()
        self.trainingFoldList.setMinimumHeight(145)
        layout.addWidget(self.trainingFoldList)
        self.trainingStageLabel = qt.QLabel("尚未开始训练")
        self.trainingStageLabel.setObjectName("homeNextStep")
        self.trainingStageLabel.setWordWrap(True)
        layout.addWidget(self.trainingStageLabel)
        self.trainingElapsedLabel = qt.QLabel("已运行时间：—")
        self.trainingElapsedLabel.setObjectName("mutedLabel")
        layout.addWidget(self.trainingElapsedLabel)

        buttonRow = qt.QHBoxLayout()
        self.trainingStartButton = self._primaryButton("开始 5 折训练")
        self.trainingStartButton.clicked.connect(lambda checked=False: self._startTraining(False))
        buttonRow.addWidget(self.trainingStartButton)
        self.trainingResumeButton = self._secondaryButton("继续未完成训练")
        self.trainingResumeButton.clicked.connect(lambda checked=False: self._startTraining(True))
        buttonRow.addWidget(self.trainingResumeButton)
        self.trainingStopButton = self._secondaryButton("停止训练")
        self.trainingStopButton.clicked.connect(self._stopActiveProcess)
        buttonRow.addWidget(self.trainingStopButton)
        self.cpuSmokeButton = self._secondaryButton("使用 CPU 测试流程")
        self.cpuSmokeButton.clicked.connect(self._runCpuSmoke)
        buttonRow.addWidget(self.cpuSmokeButton)
        layout.addLayout(buttonRow)

        self.trainingLogWidget = qt.QPlainTextEdit()
        self.trainingLogWidget.setObjectName("taskLog")
        self.trainingLogWidget.setReadOnly(True)
        self.trainingLogWidget.setMaximumHeight(180)
        self.trainingLogWidget.setPlaceholderText("原始训练日志会显示在这里…")
        layout.addWidget(self.trainingLogWidget)
        self.trainingDetailsButton = self._linkButton("查看详细日志")
        self.trainingDetailsButton.clicked.connect(self._showTrainingLogDialog)
        layout.addWidget(self.trainingDetailsButton, 0, qt.Qt.AlignLeft)
        layout.addStretch(1)
        return page

    def _buildResultsPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "实验分析",
            "实验结果",
            "这里展示真实 OOF（每个病例只由没有见过它的验证折预测）结果。没有真实预测文件时不会显示假指标。",
        )
        runRow = qt.QHBoxLayout()
        runRow.addWidget(qt.QLabel("实验记录："))
        self.resultRunSelector = qt.QComboBox()
        self.resultRunSelector.currentIndexChanged.connect(self._onResultRunChanged)
        runRow.addWidget(self.resultRunSelector, 1)
        self.refreshResultsButton = self._secondaryButton("刷新")
        self.refreshResultsButton.clicked.connect(self._refreshResultsPage)
        runRow.addWidget(self.refreshResultsButton)
        layout.addLayout(runRow)

        metricRow = qt.QHBoxLayout()
        self.resultMetricLabels = {}
        for key, caption, unit, explanation in (
            ("dice", "Dice", "", "自动分割和人工标注的重合程度，越接近 1 越好。"),
            ("iou", "IoU", "", "两个区域的重合程度，越高越好。"),
            ("hd95_mm", "HD95", " mm", "自动边界和人工边界之间的距离，越小越好。"),
        ):
            card = self._card("metricCard")
            cardLayout = qt.QVBoxLayout(card)
            cardLayout.setContentsMargins(11, 9, 11, 9)
            captionLabel = qt.QLabel(caption)
            captionLabel.setObjectName("metricCaption")
            valueLabel = qt.QLabel("暂无真实结果")
            valueLabel.setObjectName("metricValue")
            tipLabel = qt.QLabel(explanation)
            tipLabel.setObjectName("mutedLabel")
            tipLabel.setWordWrap(True)
            cardLayout.addWidget(captionLabel)
            cardLayout.addWidget(valueLabel)
            cardLayout.addWidget(tipLabel)
            metricRow.addWidget(card, 1)
            self.resultMetricLabels[key] = valueLabel
        layout.addLayout(metricRow)

        self.resultsStatusLabel = qt.QLabel("暂无实验结果。完成真实训练、OOF 预测和评价后，这里会自动更新。")
        self.resultsStatusLabel.setObjectName("resultMessage")
        self.resultsStatusLabel.setWordWrap(True)
        layout.addWidget(self.resultsStatusLabel)

        foldTitle = qt.QLabel("5 折结果")
        foldTitle.setObjectName("sectionTitle")
        layout.addWidget(foldTitle)
        self.resultFoldTable = qt.QTableWidget(0, 5)
        self.resultFoldTable.setHorizontalHeaderLabels(["分组", "病例数量", "Dice", "IoU", "HD95"])
        self.resultFoldTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.resultFoldTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.resultFoldTable.setMaximumHeight(165)
        layout.addWidget(self.resultFoldTable)

        caseTitle = qt.QLabel("病例级结果")
        caseTitle.setObjectName("sectionTitle")
        layout.addWidget(caseTitle)
        self.resultCaseTable = qt.QTableWidget(0, 5)
        self.resultCaseTable.setHorizontalHeaderLabels(["病例", "验证组", "Dice", "IoU", "HD95"])
        self.resultCaseTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.resultCaseTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.resultCaseTable.setMinimumHeight(120)
        self.resultCaseTable.currentCellChanged.connect(self._onResultCaseSelected)
        layout.addWidget(self.resultCaseTable)

        resultButtonRow = qt.QHBoxLayout()
        self.resultCompareButton = self._primaryButton("查看 GT / Prediction 对比")
        self.resultCompareButton.clicked.connect(self._showSelectedResultCase)
        resultButtonRow.addWidget(self.resultCompareButton)
        self.result3DButton = self._secondaryButton("查看 3D 对比")
        self.result3DButton.clicked.connect(lambda checked=False: self._showSelectedResultCase(show3d=True))
        resultButtonRow.addWidget(self.result3DButton)
        resultButtonRow.addWidget(qt.QLabel("显示："))
        self.resultViewModeSelector = qt.QComboBox()
        self.resultViewModeSelector.addItems(["同时显示", "人工标注", "自动预测"])
        self.resultViewModeSelector.currentIndexChanged.connect(self._applyResultViewMode)
        resultButtonRow.addWidget(self.resultViewModeSelector)
        self.resultExportButton = self._secondaryButton("导出实验结果")
        self.resultExportButton.clicked.connect(self._exportCurrentExperiment)
        resultButtonRow.addWidget(self.resultExportButton)
        resultButtonRow.addStretch(1)
        layout.addLayout(resultButtonRow)
        layout.addStretch(1)
        return page

    def _buildPredictionPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "模型应用",
            "自动分割",
            "选择一份新的匿名 MRI，系统会调用训练完成的五折 nnU-Net ensemble，生成髁突 mask 和 3D 显示。",
        )
        modelStatus = self._card("hintCard")
        modelStatusLayout = qt.QVBoxLayout(modelStatus)
        modelStatusLayout.setContentsMargins(13, 11, 13, 11)
        self.predictionModelLabel = qt.QLabel("正在检查训练模型…")
        self.predictionModelLabel.setObjectName("sectionTitle")
        self.predictionModelLabel.setWordWrap(True)
        modelStatusLayout.addWidget(self.predictionModelLabel)
        self.predictionModelHintLabel = qt.QLabel()
        self.predictionModelHintLabel.setObjectName("mutedLabel")
        self.predictionModelHintLabel.setWordWrap(True)
        modelStatusLayout.addWidget(self.predictionModelHintLabel)
        layout.addWidget(modelStatus)

        inputRow = qt.QHBoxLayout()
        inputRow.addWidget(qt.QLabel("新的 MRI："))
        self.predictionInputLabel = qt.QLabel("尚未选择")
        self.predictionInputLabel.setObjectName("mutedLabel")
        self.predictionInputLabel.setWordWrap(True)
        inputRow.addWidget(self.predictionInputLabel, 1)
        self.selectPredictionButton = self._secondaryButton("选择新的 MRI")
        self.selectPredictionButton.clicked.connect(self._choosePredictionInput)
        inputRow.addWidget(self.selectPredictionButton)
        layout.addLayout(inputRow)

        self.predictionStatusLabel = qt.QLabel("选择新的 MRI 后开始自动分割。")
        self.predictionStatusLabel.setObjectName("resultMessage")
        self.predictionStatusLabel.setWordWrap(True)
        layout.addWidget(self.predictionStatusLabel)
        buttonRow = qt.QHBoxLayout()
        self.startPredictionButton = self._primaryButton("开始自动分割")
        self.startPredictionButton.clicked.connect(self._startPrediction)
        buttonRow.addWidget(self.startPredictionButton)
        self.viewPrediction2DButton = self._secondaryButton("查看切片")
        self.viewPrediction2DButton.clicked.connect(self._showPrediction2D)
        buttonRow.addWidget(self.viewPrediction2DButton)
        self.viewPrediction3DButton = self._secondaryButton("查看 3D")
        self.viewPrediction3DButton.clicked.connect(self._showPrediction3D)
        buttonRow.addWidget(self.viewPrediction3DButton)
        self.exportPredictionButton = self._secondaryButton("导出结果")
        self.exportPredictionButton.clicked.connect(self._exportPrediction)
        buttonRow.addWidget(self.exportPredictionButton)
        buttonRow.addStretch(1)
        layout.addLayout(buttonRow)
        opacityRow = qt.QHBoxLayout()
        opacityRow.addWidget(qt.QLabel("预测透明度"))
        self.predictionOpacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.predictionOpacitySlider.setRange(20, 90)
        self.predictionOpacitySlider.setValue(55)
        self.predictionOpacitySlider.valueChanged.connect(self._onPredictionOpacityChanged)
        opacityRow.addWidget(self.predictionOpacitySlider, 1)
        self.predictionOpacityLabel = qt.QLabel("55%")
        opacityRow.addWidget(self.predictionOpacityLabel)
        layout.addLayout(opacityRow)
        self.predictionLogWidget = qt.QPlainTextEdit()
        self.predictionLogWidget.setObjectName("taskLog")
        self.predictionLogWidget.setReadOnly(True)
        self.predictionLogWidget.setMaximumHeight(150)
        layout.addWidget(self.predictionLogWidget)
        layout.addStretch(1)
        return page

    def _buildSettingsPage(self):
        page = self._card("contentCard")
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        self._addPageHeader(
            layout,
            "平台设置",
            "设置",
            "普通实验不需要修改这些选项。Slicer 路径由启动器自动管理，项目目录和技术细节只在高级功能中显示。",
        )
        settingsCard = self._card("hintCard")
        settingsLayout = qt.QFormLayout(settingsCard)
        settingsLayout.setContentsMargins(13, 11, 13, 11)
        self.settingsSlicerStatusLabel = qt.QLabel("正在检测…")
        self.settingsSlicerStatusLabel.setObjectName("sectionTitle")
        self.settingsSlicerVersionLabel = qt.QLabel("—")
        self.settingsSlicerPathLabel = qt.QLabel("路径已隐藏")
        self.settingsSlicerPathLabel.setObjectName("mutedLabel")
        self.settingsSlicerPathLabel.setWordWrap(True)
        settingsLayout.addRow("3D Slicer：", self.settingsSlicerStatusLabel)
        settingsLayout.addRow("版本：", self.settingsSlicerVersionLabel)
        settingsLayout.addRow("路径：", self.settingsSlicerPathLabel)
        slicerButtonRow = qt.QHBoxLayout()
        self.settingsChangeSlicerButton = self._secondaryButton("更换")
        self.settingsChangeSlicerButton.clicked.connect(self._changeSlicerPath)
        slicerButtonRow.addWidget(self.settingsChangeSlicerButton)
        self.settingsRedetectSlicerButton = self._secondaryButton("重新检测")
        self.settingsRedetectSlicerButton.clicked.connect(self._redetectSlicer)
        slicerButtonRow.addWidget(self.settingsRedetectSlicerButton)
        slicerButtonRow.addStretch(1)
        settingsLayout.addRow("操作：", slicerButtonRow)
        self.settingsTrainingLabel = qt.QLabel("训练环境尚未检测")
        self.settingsTrainingLabel.setWordWrap(True)
        settingsLayout.addRow("训练环境：", self.settingsTrainingLabel)
        self.settingsGpuLabel = qt.QLabel("—")
        self.settingsGpuLabel.setWordWrap(True)
        settingsLayout.addRow("显卡：", self.settingsGpuLabel)
        layout.addWidget(settingsCard)
        self.settingsInfoLabel = qt.QLabel(
            "简洁模式默认开启。患者身份不会写入实验记录；病例 ID 只允许使用匿名 case_… 标识。"
        )
        self.settingsInfoLabel.setObjectName("resultMessage")
        self.settingsInfoLabel.setWordWrap(True)
        layout.addWidget(self.settingsInfoLabel)

        advancedTitle = qt.QLabel("高级功能")
        advancedTitle.setObjectName("sectionTitle")
        layout.addWidget(advancedTitle)
        advancedRow = qt.QHBoxLayout()
        self.settingsOpenProjectButton = self._secondaryButton("打开项目目录")
        self.settingsOpenProjectButton.clicked.connect(
            lambda checked=False: self._openProjectPath(self.projectRoot)
        )
        advancedRow.addWidget(self.settingsOpenProjectButton)
        self.settingsOpenLogButton = self._secondaryButton("查看日志")
        self.settingsOpenLogButton.clicked.connect(self._openLatestLog)
        advancedRow.addWidget(self.settingsOpenLogButton)
        self.settingsOpenModelButton = self._secondaryButton("查看模型目录")
        self.settingsOpenModelButton.clicked.connect(
            lambda checked=False: self._openProjectPath(
                self.projectRoot / "workspace" / "nnUNet_results"
            )
        )
        advancedRow.addWidget(self.settingsOpenModelButton)
        advancedRow.addStretch(1)
        layout.addLayout(advancedRow)
        self.settingsAdvancedButton = self._secondaryButton("查看技术信息")
        self.settingsAdvancedButton.clicked.connect(self._showSettingsAdvanced)
        layout.addWidget(self.settingsAdvancedButton, 0, qt.Qt.AlignLeft)
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
    # Top-level experiment-platform navigation
    # ------------------------------------------------------------------
    def _showMainPage(self, pageName):
        pageName = str(pageName)
        if pageName not in {name for name, _ in MAIN_NAVIGATION}:
            return False
        if pageName != "cases" and self._mainPage == "cases" and self._dirty:
            if not self._confirmUnsaved():
                return False
        self._mainPage = pageName
        self._homeVisible = pageName == "home"
        if pageName == "home":
            self.pageStack.setCurrentIndex(PAGE_HOME)
            self._restoreWorkflowLayout()
            self._setStatusMessage("欢迎使用。首页会根据当前状态告诉你下一步做什么。", "info")
        elif pageName == "cases":
            self._currentPage = 0
            self.pageStack.setCurrentIndex(PAGE_CASE_IMPORT)
            self._refreshCaseFiles()
            self._setStatusMessage("病例与标注：导入病例、完成标注并保存。", "info")
        elif pageName == "dataset":
            self.pageStack.setCurrentIndex(PAGE_DATASET)
            self._refreshDatasetPage()
        elif pageName == "training":
            self.pageStack.setCurrentIndex(PAGE_TRAINING)
            self._refreshTrainingPage()
            if self._process is None:
                self._startEnvironmentCheck()
        elif pageName == "results":
            self.pageStack.setCurrentIndex(PAGE_RESULTS)
            self._refreshResultsPage()
        elif pageName == "prediction":
            self.pageStack.setCurrentIndex(PAGE_PREDICTION)
            self._refreshPredictionPage()
        elif pageName == "settings":
            self.pageStack.setCurrentIndex(PAGE_SETTINGS)
            self._refreshSettingsPage()
        self._updatePageChrome()
        self._syncUi()
        return True

    def _updatePageChrome(self):
        isCases = self._mainPage == "cases"
        if hasattr(self, "stepBarWidget"):
            self.stepBarWidget.setVisible(isCases)
        if hasattr(self, "caseSummaryCard"):
            self.caseSummaryCard.setVisible(isCases)
        if hasattr(self, "navigationButtons"):
            for key, button in self.navigationButtons.items():
                button.setProperty("active", key == self._mainPage)
                try:
                    button.style().unpolish(button)
                    button.style().polish(button)
                except Exception:
                    pass

    def _selectedCaseListIndex(self):
        if not hasattr(self, "caseListWidget"):
            return self._currentCaseIndex
        row = self.caseListWidget.currentRow
        if callable(row):
            row = row()
        try:
            row = int(row)
        except (TypeError, ValueError):
            row = -1
        return row if 0 <= row < len(self._caseFiles) else self._currentCaseIndex

    def _onCaseListRowChanged(self, row):
        if self._isUpdating:
            return
        try:
            row = int(row)
        except (TypeError, ValueError):
            return
        if 0 <= row < len(self._caseFiles):
            self._setStatusMessage(
                f"已选择 {self._caseIdForIndex(self._caseFiles[row], row)}，点击“继续标注”打开。",
                "info",
            )
            self._syncUi()

    def _refreshCaseListWidget(self):
        if not hasattr(self, "caseListWidget"):
            return
        selected = self._currentCaseIndex
        self._isUpdating = True
        try:
            self.caseListWidget.clear()
            for index, path in enumerate(self._caseFiles):
                caseId = self._caseIdForIndex(path, index)
                status = self._caseStatusForIndex(index)
                self.caseListWidget.addItem(
                    f"{caseId}    {self._statusSymbol(status)} {status}"
                )
            listCount = self._qtInt(self.caseListWidget, "count")
            if 0 <= selected < listCount:
                self.caseListWidget.setCurrentRow(selected)
            elif listCount:
                self.caseListWidget.setCurrentRow(0)
        finally:
            self._isUpdating = False

    def _continueSelectedCase(self):
        return self._openSelectedCase("continue")

    def _viewSelectedCase(self):
        return self._openSelectedCase("view")

    def _reeditSelectedCase(self):
        return self._openSelectedCase("reedit")

    def _openSelectedCase(self, mode="continue"):
        index = self._selectedCaseListIndex()
        if index < 0 or index >= len(self._caseFiles):
            self._setStatusMessage("请先导入或选择一个病例。", "warning")
            return False
        if not self._loadCaseAtIndex(index):
            return False
        if mode == "view" and self._caseStatusForIndex(index) in {"已标注", "已确认"}:
            self._setStatusMessage("已加载这个病例，可以查看或重新编辑标注。", "success")
        return self._startAnnotation()

    # ------------------------------------------------------------------
    # Non-blocking external workflow tasks
    # ------------------------------------------------------------------
    def _startExternalProcess(self, kind, command, *, logPath=None):
        if self._process is not None:
            self._setStatusMessage("已有任务正在运行，请等待完成或先停止当前任务。", "warning")
            return False
        try:
            process = qt.QProcess(self.rootWidget)
            environment = qt.QProcessEnvironment.systemEnvironment()
            projectRoot = str(self.projectRoot.resolve())
            # Slicer embeds its own Python.  Its PYTHONHOME/PYTHONPATH must
            # not leak into the project's external interpreter, otherwise
            # Python can mix Slicer's stdlib with the venv and fail with
            # ``SRE module mismatch`` before any project script starts.
            environment.remove("PYTHONHOME")
            environment.remove("PYTHONPATH")
            environment.insert("PYTHONPATH", projectRoot)
            environment.insert("nnUNet_raw", str((self.projectRoot / "workspace" / "nnUNet_raw").resolve()))
            environment.insert("nnUNet_preprocessed", str((self.projectRoot / "workspace" / "nnUNet_preprocessed").resolve()))
            environment.insert("nnUNet_results", str((self.projectRoot / "workspace" / "nnUNet_results").resolve()))
            process.setProcessEnvironment(environment)
            process.setWorkingDirectory(projectRoot)
            try:
                process.setProcessChannelMode(qt.QProcess.MergedChannels)
            except Exception:
                pass
            process.readyReadStandardOutput.connect(self._readProcessOutput)
            try:
                process.readyReadStandardError.connect(self._readProcessOutput)
            except Exception:
                pass
            process.finished.connect(self._onProcessFinished)
            try:
                process.errorOccurred.connect(self._onProcessError)
            except Exception:
                pass
            self._process = process
            self._processKind = str(kind)
            self._processOutput = ""
            self._processLogPath = Path(logPath) if logPath else None
            self._processStartedAt = time.monotonic()
            self._processStopRequested = False
            if self._processLogPath:
                self._processLogPath.parent.mkdir(parents=True, exist_ok=True)
                with self._processLogPath.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"\n===== {datetime.datetime.now().isoformat(timespec='seconds')} =====\n"
                        f"COMMAND: {' '.join(str(value) for value in command)}\n"
                    )
            program = str(command[0])
            arguments = [str(value) for value in command[1:]]
            process.start(program, arguments)
            if self._taskTimer is None:
                self._taskTimer = qt.QTimer(self.rootWidget)
                self._taskTimer.timeout.connect(self._updateTaskElapsed)
            self._taskTimer.start(1000)
            return True
        except Exception:
            self._process = None
            self._setDetails("启动后台任务失败\n" + traceback.format_exc())
            self._setStatusMessage("后台任务没有启动，请检查项目 Python 环境。", "warning")
            return False

    @staticmethod
    def _qprocessBytes(value):
        try:
            return bytes(value.data()).decode("utf-8", errors="replace")
        except Exception:
            try:
                return str(value)
            except Exception:
                return ""

    def _readProcessOutput(self):
        process = self._process
        if process is None:
            return
        chunks = []
        try:
            chunks.append(self._qprocessBytes(process.readAllStandardOutput()))
        except Exception:
            pass
        try:
            chunks.append(self._qprocessBytes(process.readAllStandardError()))
        except Exception:
            pass
        text = "".join(chunk for chunk in chunks if chunk)
        if not text:
            return
        self._processOutput += text
        if self._processLogPath:
            try:
                with self._processLogPath.open("a", encoding="utf-8") as handle:
                    handle.write(text)
            except Exception:
                pass
        if self._processKind == "training":
            summaryLines = []
            for line in text.splitlines():
                event = parse_training_line(line)
                if "event" in event:
                    self._trainingFoldEvents[int(event["fold"])] = event
                    if event.get("event") == "start":
                        self._activeTrainingFold = int(event["fold"])
                if event.get("event"):
                    message = user_training_message(event)
                    self.trainingStageLabel.setText(message)
                    summaryLines.append(message)
                elif event.get("epoch") is not None:
                    if event.get("fold") is None and self._activeTrainingFold is not None:
                        event = dict(event)
                        event["fold"] = self._activeTrainingFold
                    message = user_training_message(event)
                    self.trainingStageLabel.setText(message)
                    summaryLines.append(message)
            self._refreshTrainingFoldList()
            # Keep the default page readable for medical students.  The raw
            # nnU-Net stdout is retained in _processOutput/logs and is shown
            # only through “查看详细日志”.
            if summaryLines:
                self.trainingLogWidget.appendPlainText("\n".join(summaryLines))
        elif self._processKind in {"prediction", "cpu_smoke"}:
            self.predictionLogWidget.appendPlainText(text.rstrip())

    def _onProcessError(self, error):
        self._setDetails(
            f"后台任务错误\n任务：{self._processKind}\n错误：{error}\n"
            + self._processOutput[-4000:]
        )

    def _updateTaskElapsed(self):
        if self._processStartedAt is None:
            return
        elapsed = max(0, int(time.monotonic() - self._processStartedAt))
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        textValue = f"已运行时间：{hours:02d}:{minutes:02d}:{seconds:02d}"
        if hasattr(self, "trainingElapsedLabel") and self._processKind in {
            "training", "oof", "evaluation"
        }:
            self.trainingElapsedLabel.setText(textValue)

    def _stopActiveProcess(self):
        process = self._process
        if process is None:
            self._setStatusMessage("当前没有正在运行的后台任务。", "info")
            return False
        self._processStopRequested = True
        self._setStatusMessage("正在请求停止当前任务；已有 fold 的文件会保留。", "warning")
        try:
            process.terminate()
            qt.QTimer.singleShot(2500, lambda: self._killProcessIfRunning(process))
        except Exception:
            self._killProcessIfRunning(process)
        return True

    def _killProcessIfRunning(self, process):
        if self._process is not process:
            return
        try:
            if process.state() != qt.QProcess.NotRunning:
                process.kill()
        except Exception:
            pass

    def _onProcessFinished(self, exitCode, *args):
        process = self._process
        kind = self._processKind
        try:
            exitCode = int(exitCode)
        except (TypeError, ValueError):
            exitCode = 1
        stopped = self._processStopRequested
        self._readProcessOutput()
        if self._taskTimer is not None:
            self._taskTimer.stop()
        self._process = None
        self._processKind = ""
        self._processStartedAt = None
        self._processStopRequested = False
        if stopped:
            if kind in {"training", "oof", "evaluation"} and self._currentRunDir:
                finalize_experiment_run(
                    self._currentRunDir,
                    summary={"status": "stopped", "completed_folds": list(self._trainingFoldEvents)},
                )
            self._setStatusMessage("任务已停止。已完成的 fold 会在下次点击“继续未完成训练”时保留。", "warning")
            self._refreshTrainingPage()
            return
        if kind == "environment":
            self._finishEnvironmentCheck(exitCode)
        elif kind == "dataset_validate":
            self._finishDatasetValidation(exitCode)
        elif kind == "dataset_build":
            self._finishDatasetBuild(exitCode)
        elif kind == "training":
            self._finishTraining(exitCode)
        elif kind == "oof":
            self._finishOof(exitCode)
        elif kind == "evaluation":
            self._finishEvaluation(exitCode)
        elif kind == "prediction":
            self._finishPrediction(exitCode)
        elif kind == "cpu_smoke":
            self._finishCpuSmoke(exitCode)

    def _startEnvironmentCheck(self):
        if self._process is not None:
            return False
        self.trainingEnvironmentLabel.setText("正在检查 Python、nnU-Net、训练数据和显卡…")
        return self._startExternalProcess(
            "environment",
            environment_command(project_root=self.projectRoot, python_executable=project_python_executable(self.projectRoot)),
        )

    def _startDatasetValidation(self):
        if self._process is not None:
            self._setStatusMessage("已有任务正在运行，请等待完成。", "warning")
            return False
        self._validationRows = []
        self._validationPassed = False
        self._datasetPrepared = False
        self._setDatasetSteps(0)
        self._setResultMessage(self.datasetResultLabel, "正在检查全部病例，请稍候…", "neutral")
        return self._startExternalProcess(
            "dataset_validate",
            dataset_validation_command(
                project_root=self.projectRoot,
                python_executable=project_python_executable(self.projectRoot),
            ),
        )

    def _startDatasetBuild(self):
        if not self._validationPassed:
            self._setResultMessage(self.datasetResultLabel, "请先检查全部病例；检查通过后才能准备训练数据。", "warning")
            return False
        self._setDatasetSteps(1)
        self._setResultMessage(self.datasetResultLabel, "正在整理训练数据并生成 5 组实验…", "neutral")
        return self._startExternalProcess(
            "dataset_build",
            dataset_build_command(
                project_root=self.projectRoot,
                python_executable=project_python_executable(self.projectRoot),
            ),
        )

    def _finishEnvironmentCheck(self, exitCode):
        try:
            self._environmentReport = parse_environment_json(self._processOutput)
        except Exception:
            self._environmentReport = None
            self._setDetails("环境检查没有返回可读报告\n" + self._processOutput[-6000:])
        if self._environmentReport:
            display = environment_display(self._environmentReport)
            self.trainingEnvironmentLabel.setText(
                "训练环境：nnU-Net　{nnunet} {nnunet_text}\n"
                "训练数据：{data_text}\n"
                "显卡：{gpu_message}".format(
                    **display,
                    nnunet_text="已安装" if display["nnunet"] == "✓" else "未安装",
                    data_text="已准备" if display["data"] == "✓" else "未准备",
                )
            )
            if display["gpu"] != "✓":
                self.trainingGpuWarningLabel.setText(
                    "当前电脑没有检测到可用于训练的 NVIDIA 显卡。\n"
                    "可以使用“CPU 测试流程”检查软件连接，或稍后在有兼容显卡的电脑正式训练。"
                )
            else:
                self.trainingGpuWarningLabel.clear()
            self._setDetails(
                "环境检查（高级信息）\n"
                + json.dumps(self._environmentReport, ensure_ascii=False, indent=2)
            )
        else:
            self.trainingEnvironmentLabel.setText("环境检查失败，请打开高级信息查看原始日志。")
            self.trainingGpuWarningLabel.setText("无法确认当前电脑是否适合正式训练。")
        self._refreshTrainingPage()
        if self._mainPage == "settings":
            self._refreshSettingsPage()
        if exitCode == 0 and self._environmentReport:
            self._setStatusMessage("系统检查完成。", "success")
        else:
            self._setStatusMessage("系统检查没有完整通过，请查看训练页提示。", "warning")

    def _finishDatasetValidation(self, exitCode):
        self._validationRows = read_validation_csv(self.projectRoot / "workspace" / "reports" / "dataset_validation.csv")
        inventory, _ = self._inventoryAndCounts()
        self._validationPassed = (
            exitCode == 0
            and self._validationPassesVerifiedInventory(inventory)
        )
        self._setDatasetSteps(1 if self._validationPassed else 0)
        self._refreshDatasetPage()
        if self._validationPassed:
            self._setResultMessage(
                self.datasetResultLabel,
                f"✓ {len(self._validationRows)} 个病例可以用于训练。下一步可以准备训练数据。",
                "success",
            )
            self._setStatusMessage("病例检查通过，可以准备训练数据。", "success")
        else:
            failed = [row for row in self._validationRows if row.get("status") == "FAIL"]
            details = [f"发现 {len(failed)} 个问题："]
            for row in failed[:12]:
                issue = row.get("errors") or "需要人工确认"
                details.append(f"{row.get('case_id', '未知病例')}：{issue}")
            if not failed:
                details.append("没有找到可用于训练的已确认病例。")
            self._setResultMessage(self.datasetResultLabel, "\n".join(details), "warning")
            self._setStatusMessage("数据检查未通过，正式实验暂时不能开始。", "warning")

    def _finishDatasetBuild(self, exitCode):
        datasetPath = self.projectRoot / "workspace" / "nnUNet_raw" / "Dataset501_CondyleMRI" / "dataset.json"
        splitPath = self.projectRoot / "workspace" / "nnUNet_preprocessed" / "Dataset501_CondyleMRI" / "splits_final.json"
        try:
            splitCount = len(read_splits(splitPath)) if splitPath.exists() else 0
        except Exception:
            splitCount = 0
        inventory, counts = self._inventoryAndCounts()
        self._validationPassed = self._validationPassesVerifiedInventory(inventory)
        self._datasetPrepared = (
            exitCode == 0
            and datasetPath.exists()
            and splitCount == len(FOLDS)
            and self._validationPassed
            and int(counts.get("trainable", 0)) >= len(FOLDS)
        )
        self._preprocessingPrepared = False
        if self._datasetPrepared:
            self._setDatasetSteps(3)
            self._setResultMessage(
                self.datasetResultLabel,
                "✓ 训练数据准备完成\n已生成真实病例数据和 grouped 5-fold 分组，可以进入模型训练。",
                "success",
            )
            self._setStatusMessage("训练数据准备完成，可以进入模型训练。", "success")
        else:
            self._setResultMessage(
                self.datasetResultLabel,
                "训练数据没有准备完成，请查看高级信息和详细日志。",
                "warning",
            )
            self._setStatusMessage("训练数据准备失败，正式训练暂时不能开始。", "warning")
        self._setDetails(
            f"数据准备结果\nDataset：{datasetPath}\nsplits：{splitPath}\n"
            f"fold 数量：{splitCount}\n原始输出：\n{self._processOutput[-5000:]}"
        )
        self._refreshDatasetPage()
        self._refreshTrainingPage()

    def _setDatasetSteps(self, completed):
        if not hasattr(self, "datasetStepLabels"):
            return
        try:
            completed = int(completed)
        except (TypeError, ValueError):
            completed = 0
        for index, label in enumerate(self.datasetStepLabels):
            if index < completed:
                label.setText("✓ " + ("检查病例", "整理训练数据", "生成 5 组实验")[index])
            elif index == completed:
                label.setText("● " + ("检查病例", "整理训练数据", "生成 5 组实验")[index])
            else:
                label.setText("○ " + ("检查病例", "整理训练数据", "生成 5 组实验")[index])

    def _inventoryAndCounts(self):
        try:
            inventory = load_case_inventory(
                manifest_path=self.manifestPath,
                images_dir=self.niftiDir,
                labels_dir=self.labelsDir,
            )
        except Exception as exc:
            inventory = []
            self._setDetails(f"病例统计失败\n{type(exc).__name__}: {exc}")
        return inventory, case_counts(inventory, validation_rows=self._validationRows)

    def _refreshDerivedProjectState(self, inventory, counts):
        """Re-read persisted dataset state so the homepage is not stale."""

        self._validationPassed = self._validationPassesVerifiedInventory(inventory)
        datasetPath = (
            self.projectRoot
            / "workspace"
            / "nnUNet_raw"
            / "Dataset501_CondyleMRI"
            / "dataset.json"
        )
        splitPath = (
            self.projectRoot
            / "workspace"
            / "nnUNet_preprocessed"
            / "Dataset501_CondyleMRI"
            / "splits_final.json"
        )
        splitCount = 0
        if splitPath.exists():
            try:
                splitCount = len(read_splits(splitPath))
            except Exception:
                splitCount = 0
        self._datasetPrepared = bool(
            datasetPath.exists()
            and splitCount == len(FOLDS)
            and self._validationPassed
            and int(counts.get("trainable", 0)) >= len(FOLDS)
        )
        fingerprintPath = (
            self.projectRoot
            / "workspace"
            / "nnUNet_preprocessed"
            / "Dataset501_CondyleMRI"
            / "dataset_fingerprint.json"
        )
        plansPath = (
            self.projectRoot
            / "workspace"
            / "nnUNet_preprocessed"
            / "Dataset501_CondyleMRI"
            / "nnUNetPlans.json"
        )
        self._preprocessingPrepared = bool(
            self._datasetPrepared
            and fingerprintPath.exists()
            and plansPath.exists()
        )

    def _validationPassesVerifiedInventory(self, inventory=None):
        """Reject stale or non-VERIFIED reports before enabling training."""

        rows = list(inventory) if inventory is not None else self._inventoryAndCounts()[0]
        verified_ids = {
            str(item.get("case_id"))
            for item in rows
            if bool(item.get("verified"))
        }
        report_ids = {
            str(item.get("case_id"))
            for item in self._validationRows
        }
        if not verified_ids or report_ids != verified_ids:
            return False
        return all(
            str(item.get("status")) == "PASS"
            and str(item.get("annotation_status", "")).upper() == "VERIFIED"
            for item in self._validationRows
        )

    def _refreshDatasetPage(self):
        if not hasattr(self, "datasetSummaryLabel"):
            return
        inventory, counts = self._inventoryAndCounts()
        self._validationPassed = self._validationPassesVerifiedInventory(inventory)
        annotated = int(counts["annotated"])
        verified = int(counts.get("verified", 0))
        trainable = int(counts["trainable"])
        issues = int(counts["issues"])
        self.datasetSummaryLabel.setText(
            f"病例总数：{counts['total']}　　人工标注完成：{annotated}\n"
            f"已验证标注：{verified}　　可用于训练：{trainable}　　检查问题：{issues}"
        )
        self.datasetGuidanceLabel.setText(
            count_guidance(int(counts.get("verified_group_count", 0)))
            + "\n只有状态为 VERIFIED（已确认）的病例会进入正式训练；同一患者的左右侧会自动放在同一折。"
        )
        excluded = int(counts["total"]) - verified
        self.datasetExcludedLabel.setText(
            f"未确认病例：{excluded}　这些病例只用于提醒，不会被复制到 nnU-Net 训练数据。"
            if excluded
            else "目前所有病例都已确认。"
        )
        self.datasetCaseTable.setRowCount(0)
        validationByCase = {str(row.get("case_id")): row for row in self._validationRows}
        for item in inventory:
            if str(item.get("annotation_status", "")).upper() != "VERIFIED":
                continue
            rowIndex = self._qtInt(self.datasetCaseTable, "rowCount")
            self.datasetCaseTable.insertRow(rowIndex)
            validation = validationByCase.get(str(item.get("case_id")), {})
            values = (
                item.get("case_id", ""),
                item.get("status", ""),
                item.get("group_id", ""),
                validation.get("errors") or ", ".join(item.get("problems", [])) or "—",
            )
            for column, value in enumerate(values):
                self.datasetCaseTable.setItem(rowIndex, column, qt.QTableWidgetItem(str(value)))
        self.buildDatasetButton.setEnabled(self._validationPassed and self._process is None)
        self.checkDatasetButton.setEnabled(self._process is None)
        self.datasetAdvancedLabel.setText(
            "高级信息：\n"
            f"Dataset501_CondyleMRI：{self.projectRoot / 'workspace' / 'nnUNet_raw' / 'Dataset501_CondyleMRI'}\n"
            f"splits_final.json：{self.projectRoot / 'workspace' / 'nnUNet_preprocessed' / 'Dataset501_CondyleMRI' / 'splits_final.json'}"
        )

    def _refreshTrainingFoldList(self):
        if not hasattr(self, "trainingFoldList"):
            return
        states = detect_fold_states(fold_results_directory(results_root=self.projectRoot / "workspace" / "nnUNet_results"))
        self.trainingFoldList.clear()
        for state in states:
            event = self._trainingFoldEvents.get(state.fold, {})
            eventName = event.get("event")
            if eventName == "start":
                status = "训练中"
                symbol = "●"
            elif eventName == "failed":
                status = "未完成"
                symbol = "!"
            elif eventName == "complete" or state.completed:
                status = "完成"
                symbol = "✓"
            elif state.resumable:
                status = "可继续"
                symbol = "●"
            else:
                status = "等待"
                symbol = "○"
            self.trainingFoldList.addItem(f"第 {state.fold + 1} 组 / Fold {state.fold + 1}    {symbol} {status}")

    def _refreshTrainingPage(self):
        if not hasattr(self, "trainingStartButton"):
            return
        if not self._validationRows:
            self._validationRows = read_validation_csv(self.projectRoot / "workspace" / "reports" / "dataset_validation.csv")
        inventory, counts = self._inventoryAndCounts()
        self._validationPassed = self._validationPassesVerifiedInventory(inventory)
        splitPath = self.projectRoot / "workspace" / "nnUNet_preprocessed" / "Dataset501_CondyleMRI" / "splits_final.json"
        datasetPath = self.projectRoot / "workspace" / "nnUNet_raw" / "Dataset501_CondyleMRI" / "dataset.json"
        fingerprintPath = self.projectRoot / "workspace" / "nnUNet_preprocessed" / "Dataset501_CondyleMRI" / "dataset_fingerprint.json"
        plansPath = self.projectRoot / "workspace" / "nnUNet_preprocessed" / "Dataset501_CondyleMRI" / "nnUNetPlans.json"
        self._datasetPrepared = bool(
            datasetPath.exists()
            and splitPath.exists()
            and self._validationPassed
            and int(counts.get("trainable", 0)) >= len(FOLDS)
        )
        self._preprocessingPrepared = bool(
            self._datasetPrepared
            and fingerprintPath.exists()
            and plansPath.exists()
        )
        cudaReady = bool(
            self._environmentReport
            and isinstance(self._environmentReport.get("cuda"), dict)
            and self._environmentReport["cuda"].get("status") == "PASS"
        )
        envReady = bool(self._environmentReport and self._environmentReport.get("nnunet_ready"))
        readiness = assess_training_readiness(
            annotated_cases=int(counts.get("verified", 0)),
            group_count=int(counts.get("verified_group_count", 0)),
            validation_passed=self._validationPassed,
            environment_ready=envReady,
            gpu_ready=cudaReady,
            dataset_prepared=self._datasetPrepared,
        )
        self.trainingReadinessLabel.setText(
            ("✓ " if readiness.formal_ready else "") + readiness.message
            + ("\n" + "\n".join(readiness.reasons) if readiness.reasons and not readiness.formal_ready else "")
        )
        self._refreshTrainingFoldList()
        states = detect_fold_states(fold_results_directory(results_root=self.projectRoot / "workspace" / "nnUNet_results"))
        incomplete = any(state.resumable for state in states)
        running = self._process is not None
        self.trainingStartButton.setEnabled(readiness.formal_ready and not running)
        self.trainingResumeButton.setEnabled(readiness.formal_ready and incomplete and not running)
        self.trainingStopButton.setEnabled(running and self._processKind in {"training", "oof", "evaluation"})
        self.cpuSmokeButton.setEnabled(readiness.pipeline_ready and not running)
        self.trainingDetailsButton.setEnabled(bool(self._processOutput or self._currentRunDir))
        if not running and self._processKind == "":
            if readiness.level == "blocked_gpu":
                self.trainingStageLabel.setText("当前机器暂不适合正式训练。可以先运行 CPU 流程检查。")
            elif readiness.formal_ready:
                self.trainingStageLabel.setText("可以开始正式 5 折训练。")
            elif not counts["annotated"]:
                self.trainingStageLabel.setText("当前还没有足够的人工标注病例，无法进行正式实验。")

    def _ensureExperimentRun(self, resume=False):
        if self._currentRunDir and self._currentRunDir.exists():
            return self._currentRunDir
        _, counts = self._inventoryAndCounts()
        cuda = self._environmentReport.get("cuda", {}) if self._environmentReport else {}
        packages = self._environmentReport.get("packages", {}) if self._environmentReport else {}
        modelPath = fold_results_directory(
            results_root=self.projectRoot / "workspace" / "nnUNet_results"
        )
        self._currentRunDir = create_experiment_run(
            workspace_dir=self.projectRoot / "workspace",
            config={
                "case_count": counts.get("total", 0),
                "annotated_count": counts.get("annotated", 0),
                "verified_count": counts.get("verified", 0),
                "group_count": counts.get("verified_group_count", 0),
                "validation_passed": self._validationPassed,
                "device": "cuda",
                "gpu": cuda.get("device_name", "unavailable"),
                "nnunet_version": packages.get("nnunetv2", "unknown"),
                "configuration": "3d_fullres",
                "folds": list(FOLDS),
                "model_path": str(modelPath),
                "resume": bool(resume),
                "model": "nnU-Net v2 3d_fullres",
            },
        )
        return self._currentRunDir

    def _startTraining(self, resume=False):
        if self._process is not None:
            return False
        if not self._environmentReport:
            self._setStatusMessage("正在检查训练环境，请检查完成后再开始训练。", "info")
            self._startEnvironmentCheck()
            return False
        self._refreshTrainingPage()
        _, counts = self._inventoryAndCounts()
        cudaReady = bool(
            isinstance(self._environmentReport.get("cuda"), dict)
            and self._environmentReport["cuda"].get("status") == "PASS"
        )
        readiness = assess_training_readiness(
            annotated_cases=int(counts.get("verified", 0)),
            group_count=int(counts.get("verified_group_count", 0)),
            validation_passed=self._validationPassed,
            environment_ready=bool(self._environmentReport.get("nnunet_ready")),
            gpu_ready=cudaReady,
            dataset_prepared=self._datasetPrepared,
        )
        if not readiness.formal_ready:
            self._setStatusMessage(readiness.message, "warning")
            return False
        summary = training_prerequisite_summary(
            available_cases=int(counts.get("verified", 0)),
            patient_groups=int(counts.get("verified_group_count", 0)),
            validation_passed=self._validationPassed,
            environment_ready=bool(self._environmentReport.get("nnunet_ready")),
            gpu_ready=cudaReady,
            dataset_prepared=self._datasetPrepared,
        )
        confirm = qt.QMessageBox(slicer.util.mainWindow())
        confirm.setIcon(qt.QMessageBox.Information)
        confirm.setWindowTitle("开始 5 折训练")
        confirm.setText(str(summary["text"]))
        beginButton = confirm.addButton("开始", qt.QMessageBox.AcceptRole)
        cancelButton = confirm.addButton("取消", qt.QMessageBox.RejectRole)
        confirm.setDefaultButton(cancelButton)
        self._execDialog(confirm)
        if confirm.clickedButton() != beginButton:
            self._setStatusMessage("已取消训练。", "info")
            return False
        runDir = self._ensureExperimentRun(resume=resume)
        self._trainingFoldEvents = {}
        self._activeTrainingFold = None
        self.trainingLogWidget.clear()
        self.trainingStageLabel.setText("正在准备数据和训练任务…")
        command = training_command(
            device="cuda",
            resume=bool(resume),
            plan=not self._preprocessingPrepared,
            project_root=self.projectRoot,
            python_executable=project_python_executable(self.projectRoot),
        )
        started = self._startExternalProcess(
            "training",
            command,
            logPath=runDir / "logs" / "training.log",
        )
        if started:
            self._setStatusMessage(
                "正式 5 折训练已在后台开始，Slicer 仍然可以操作。",
                "success",
            )
            self._refreshTrainingPage()
        return started

    def _finishTraining(self, exitCode):
        self._refreshTrainingPage()
        if exitCode != 0:
            self.trainingStageLabel.setText("训练没有完成；已完成的 fold 会保留，可以继续未完成训练。")
            self._setStatusMessage("训练任务未完成，请查看详细日志。", "warning")
            return
        self._preprocessingPrepared = True
        self.trainingStageLabel.setText("5 折训练完成，正在生成真实 OOF 预测…")
        self._setStatusMessage("训练完成，正在生成验证集预测。", "success")
        self._startExternalProcess(
            "oof",
            oof_command(
                device="cuda",
                project_root=self.projectRoot,
                python_executable=project_python_executable(self.projectRoot),
            ),
            logPath=(self._currentRunDir / "logs" / "training.log") if self._currentRunDir else None,
        )

    def _finishOof(self, exitCode):
        if exitCode != 0:
            self.trainingStageLabel.setText("OOF 预测没有完成，实验结果暂不可用。")
            self._setStatusMessage("真实 OOF 预测失败，没有生成假指标。", "warning")
            return
        self.trainingStageLabel.setText("OOF 预测完成，正在计算 Dice / IoU / HD95…")
        self._startExternalProcess(
            "evaluation",
            evaluation_command(
                project_root=self.projectRoot,
                python_executable=project_python_executable(self.projectRoot),
            ),
            logPath=(self._currentRunDir / "logs" / "training.log") if self._currentRunDir else None,
        )

    def _finishEvaluation(self, exitCode):
        if exitCode != 0 or not has_evaluation_results(self.projectRoot / "workspace" / "reports"):
            self.trainingStageLabel.setText("评价没有完成，实验结果暂不可用。")
            self._setStatusMessage("评价失败，没有显示假指标。", "warning")
            return
        summary = read_metrics_summary(self.projectRoot / "workspace" / "reports" / "metrics_summary.csv")
        rows = read_metrics_csv(self.projectRoot / "workspace" / "reports" / "metrics_per_case.csv")
        if self._currentRunDir:
            finalize_experiment_run(
                self._currentRunDir,
                summary={
                    "status": "complete",
                    "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "case_count": len(rows),
                    "metrics": summary,
                    "completed_folds": list(FOLDS),
                },
            )
        self.trainingStageLabel.setText("训练、OOF 预测和评价全部完成。")
        self._setStatusMessage("实验完成，可以查看 Dice / IoU / HD95。", "success")
        self._refreshResultsPage()
        self._refreshTrainingPage()

    def _runCpuSmoke(self):
        if self._process is not None:
            return False
        self._refreshTrainingPage()
        _, counts = self._inventoryAndCounts()
        if not self._validationPassed or int(counts.get("verified_group_count", 0)) < len(FOLDS):
            self._setStatusMessage("请先准备至少 5 个通过检查的不同患者组。", "warning")
            return False
        self.trainingStageLabel.setText("正在运行 CPU 流程检查（不会训练模型，也不会生成指标）…")
        command = script_command(
            "pipeline_smoke.py",
            project_root=self.projectRoot,
            python_executable=project_python_executable(self.projectRoot),
        )
        return self._startExternalProcess("cpu_smoke", command)

    def _finishCpuSmoke(self, exitCode):
        if exitCode == 0:
            self.trainingStageLabel.setText("CPU 流程检查通过；没有运行正式训练，也没有生成实验指标。")
            self._setStatusMessage("CPU 流程检查完成。正式结果仍需要兼容 CUDA GPU。", "success")
        else:
            self.trainingStageLabel.setText("CPU 流程检查未通过，请查看日志。")
            self._setStatusMessage("CPU 流程检查失败。", "warning")

    # ------------------------------------------------------------------
    # Experiment results and history
    # ------------------------------------------------------------------
    @staticmethod
    def _comboData(combo, index):
        try:
            value = combo.itemData(index)
            if callable(value):
                value = value()
            return value
        except Exception:
            return None

    @staticmethod
    def _qtInt(obj, attribute, default=0):
        try:
            value = getattr(obj, attribute)
            if callable(value):
                value = value()
            return int(value)
        except (AttributeError, TypeError, ValueError):
            return int(default)

    @staticmethod
    def _resolvePath(value, root):
        if not value:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = Path(root) / path
        return path.resolve()

    def _refreshResultsPage(self):
        if not hasattr(self, "resultRunSelector"):
            return
        current = str(self._resultRunDir) if self._resultRunDir else ""
        self.resultRunSelector.blockSignals(True)
        try:
            self.resultRunSelector.clear()
            runs = list_experiment_runs(self.projectRoot / "workspace")
            for run in runs:
                record = read_experiment_record(run)
                summary = record.get("summary", {}) if isinstance(record, dict) else {}
                metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
                dice = metrics.get("dice", {}) if isinstance(metrics, dict) else {}
                mean = dice.get("mean") if isinstance(dice, dict) else None
                label = run.name
                if mean is not None:
                    label += f"　Dice {format_metric(mean)}"
                self.resultRunSelector.addItem(label, str(run))
            reportRoot = self.projectRoot / "workspace" / "reports"
            if has_evaluation_results(reportRoot):
                self.resultRunSelector.addItem("当前报告（未归档实验）", "__reports__")
            comboCount = self._qtInt(self.resultRunSelector, "count")
            if comboCount == 0:
                self.resultRunSelector.addItem("暂无真实实验记录", "")
            chosen = -1
            for index in range(self._qtInt(self.resultRunSelector, "count")):
                if str(self._comboData(self.resultRunSelector, index) or "") == current:
                    chosen = index
                    break
            if chosen >= 0:
                self.resultRunSelector.setCurrentIndex(chosen)
            elif current:
                self.resultRunSelector.setCurrentIndex(0)
        finally:
            self.resultRunSelector.blockSignals(False)
        index = self.resultRunSelector.currentIndex
        if callable(index):
            index = index()
        self._loadResultSource(self._comboData(self.resultRunSelector, int(index)))

    def _onResultRunChanged(self, index):
        self._loadResultSource(self._comboData(self.resultRunSelector, int(index)))

    def _loadResultSource(self, source):
        if source and str(source) != "__reports__":
            sourcePath = Path(str(source))
            self._resultRunDir = sourcePath if sourcePath.exists() else None
        else:
            self._resultRunDir = None
        sourceRoot = self._resultRunDir or (self.projectRoot / "workspace" / "reports")
        summaryPath = sourceRoot / "metrics_summary.csv"
        casePath = sourceRoot / "metrics_per_case.csv"
        self._resultSummary = read_metrics_summary(summaryPath)
        self._resultRows = read_metrics_csv(casePath)
        self._updateResultsWidgets(sourceRoot)

    def _updateResultsWidgets(self, sourceRoot):
        for key in ("dice", "iou", "hd95_mm"):
            row = self._resultSummary.get(key, {})
            unit = " mm" if key == "hd95_mm" else ""
            self.resultMetricLabels[key].setText(
                format_metric(row.get("mean"), row.get("std"), unit=unit)
            )
        if not self._resultRows:
            self.resultsStatusLabel.setText(
                "暂无真实实验结果。完成真实训练、OOF 预测和评价后，这里会显示病例级指标。"
            )
        else:
            self.resultsStatusLabel.setText(
                f"已读取 {len(self._resultRows)} 个真实 OOF 病例结果。"
                "预测来自对应验证折，不是训练集结果。"
            )
        self.resultFoldTable.setRowCount(0)
        for row in summarize_metrics_by_fold(self._resultRows):
            rowIndex = self._qtInt(self.resultFoldTable, "rowCount")
            self.resultFoldTable.insertRow(rowIndex)
            values = (
                f"第 {int(row['fold']) + 1} 组 / Fold {int(row['fold']) + 1}",
                row.get("case_count", 0),
                format_metric(row.get("dice_mean"), row.get("dice_std")),
                format_metric(row.get("iou_mean"), row.get("iou_std")),
                format_metric(row.get("hd95_mm_mean"), row.get("hd95_mm_std"), unit=" mm"),
            )
            for column, value in enumerate(values):
                self.resultFoldTable.setItem(rowIndex, column, qt.QTableWidgetItem(str(value)))
        self.resultCaseTable.setRowCount(0)
        for row in self._resultRows:
            rowIndex = self._qtInt(self.resultCaseTable, "rowCount")
            self.resultCaseTable.insertRow(rowIndex)
            values = (
                row.get("case_id", ""),
                f"第 {int(row.get('fold', 0)) + 1} 组" if str(row.get("fold", "")).isdigit() else "—",
                format_metric(row.get("dice")),
                format_metric(row.get("iou")),
                format_metric(row.get("hd95_mm"), unit=" mm"),
            )
            for column, value in enumerate(values):
                self.resultCaseTable.setItem(rowIndex, column, qt.QTableWidgetItem(str(value)))
        self.resultCompareButton.setEnabled(bool(self._resultRows))
        self.result3DButton.setEnabled(bool(self._resultRows))
        self.resultExportButton.setEnabled(bool(self._resultRunDir))
        self._setDetails(
            f"实验结果来源：{sourceRoot}\n"
            f"metrics_summary.csv：{sourceRoot / 'metrics_summary.csv'}\n"
            f"metrics_per_case.csv：{sourceRoot / 'metrics_per_case.csv'}"
        )

    def _onResultCaseSelected(self, row, *args):
        try:
            index = int(row)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._resultRows):
            self._selectedResultCase = self._resultRows[index]

    def _selectedResultRow(self):
        if self._selectedResultCase:
            return self._selectedResultCase
        row = self.resultCaseTable.currentRow
        if callable(row):
            row = row()
        try:
            row = int(row)
        except (TypeError, ValueError):
            row = 0
        if 0 <= row < len(self._resultRows):
            self._selectedResultCase = self._resultRows[row]
            return self._selectedResultCase
        return self._resultRows[0] if self._resultRows else None

    def _showSelectedResultCase(self, show3d=False):
        row = self._selectedResultRow()
        if not row:
            self._setStatusMessage("当前没有可查看的病例级实验结果。", "warning")
            return False
        caseId = str(row.get("case_id", ""))
        fold = int(row.get("fold", 0))
        manifestRows = read_manifest(self.manifestPath)
        manifestRow = next((item for item in manifestRows if item.get("case_id") == caseId), None)
        if manifestRow is None:
            self._setStatusMessage("这个病例不在匿名 manifest 中，无法显示对比。", "warning")
            return False
        imagePath = self._resolvePath(manifestRow.get("image_path"), self.projectRoot)
        labelPath = self._resolvePath(manifestRow.get("label_path"), self.projectRoot)
        predictionPath = (
            self.projectRoot / "workspace" / "predictions" / "oof" / f"fold_{fold}" / f"{caseId}.nii.gz"
        ).resolve()
        if not imagePath or not labelPath or not imagePath.exists() or not labelPath.exists() or not predictionPath.exists():
            self._setStatusMessage("GT 或真实 OOF Prediction 文件缺失，不能显示对比。", "warning")
            return False
        try:
            self._loadResultComparison(imagePath, labelPath, predictionPath)
            if show3d:
                self._showResult3D()
            else:
                self._showResult2D()
            self._setStatusMessage(
                f"已显示 {caseId}：人工标注为绿色，模型预测为蓝色。", "success"
            )
            return True
        except Exception:
            self._setDetails("加载 GT / Prediction 对比失败\n" + traceback.format_exc())
            self._setStatusMessage("对比视图没有成功加载，请查看高级信息。", "warning")
            return False

    def _removeNodeSafely(self, node):
        if not node:
            return
        try:
            slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass

    def _clearResultComparison(self):
        for name in (
            "_resultImageNode",
            "_resultGroundTruthNode",
            "_resultPredictionNode",
            "_resultCompareSegmentationNode",
        ):
            node = getattr(self, name, None)
            self._removeNodeSafely(node)
            setattr(self, name, None)

    def _loadResultComparison(self, imagePath, labelPath, predictionPath):
        self._clearResultComparison()
        loaded = slicer.util.loadVolume(str(imagePath), returnNode=True)
        self._resultImageNode = loaded[1] if isinstance(loaded, tuple) else loaded
        gtLoaded = slicer.util.loadLabelVolume(str(labelPath), returnNode=True)
        predLoaded = slicer.util.loadLabelVolume(str(predictionPath), returnNode=True)
        gtNode = gtLoaded[1] if isinstance(gtLoaded, tuple) else gtLoaded
        predNode = predLoaded[1] if isinstance(predLoaded, tuple) else predLoaded
        if not self._resultImageNode or not gtNode or not predNode:
            raise RuntimeError("Slicer could not load image/label/prediction")
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "TMJ_GT_Prediction_Comparison")
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(self._resultImageNode)
        logic = slicer.modules.segmentations.logic()
        if logic.ImportLabelmapToSegmentationNode(gtNode, seg) is False:
            raise RuntimeError("could not import ground truth label")
        if logic.ImportLabelmapToSegmentationNode(predNode, seg) is False:
            raise RuntimeError("could not import prediction label")
        segmentation = seg.GetSegmentation()
        if segmentation.GetNumberOfSegments() < 2:
            raise RuntimeError("comparison requires two non-empty segments")
        gtId = segmentation.GetNthSegmentID(0)
        predId = segmentation.GetNthSegmentID(1)
        segmentation.GetSegment(gtId).SetName("人工标注")
        segmentation.GetSegment(gtId).SetColor(0.15, 0.8, 0.3)
        segmentation.GetSegment(predId).SetName("模型预测")
        segmentation.GetSegment(predId).SetColor(0.15, 0.4, 0.95)
        self._resultCompareSegmentationNode = seg
        self._removeNodeSafely(gtNode)
        self._removeNodeSafely(predNode)
        self._setResultDisplaySettings()
        self._showVolumeInSliceViews(self._resultImageNode)

    def _setResultDisplaySettings(self):
        seg = self._resultCompareSegmentationNode
        if not seg:
            return
        display = seg.GetDisplayNode()
        if not display:
            return
        try:
            display.SetVisibility(True)
            display.SetVisibility2DFill(True)
            display.SetVisibility2DOutline(True)
            display.SetVisibility3D(True)
            display.SetOpacity(0.45)
            if hasattr(display, "SetOpacity3D"):
                display.SetOpacity3D(0.7)
            for index in range(seg.GetSegmentation().GetNumberOfSegments()):
                segmentId = seg.GetSegmentation().GetNthSegmentID(index)
                if hasattr(display, "SetSegmentVisibility"):
                    display.SetSegmentVisibility(segmentId, True)
                if hasattr(display, "SetSegmentVisibility3D"):
                    display.SetSegmentVisibility3D(segmentId, True)
                if hasattr(display, "SetSegmentOpacity"):
                    display.SetSegmentOpacity(segmentId, 0.45)
                if hasattr(display, "SetSegmentOpacity3D"):
                    display.SetSegmentOpacity3D(segmentId, 0.7)
        except Exception:
            pass

    def _showResult2D(self):
        if not self._resultCompareSegmentationNode or not self._resultImageNode:
            return False
        self._setCheckLayout()
        self._showVolumeInSliceViews(self._resultImageNode)
        self._setResultDisplaySettings()
        self._setDetails(self._lastDetailText + "\n2D 图例：人工标注=绿色；模型预测=蓝色")
        return True

    def _applyResultViewMode(self, index):
        seg = self._resultCompareSegmentationNode
        if not seg:
            return
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        display = seg.GetDisplayNode()
        if not display:
            return
        for segmentIndex in range(seg.GetSegmentation().GetNumberOfSegments()):
            segmentId = seg.GetSegmentation().GetNthSegmentID(segmentIndex)
            visible = index == 0 or (index == 1 and segmentIndex == 0) or (index == 2 and segmentIndex == 1)
            try:
                display.SetSegmentVisibility(segmentId, visible)
                display.SetSegmentVisibility3D(segmentId, visible)
            except Exception:
                pass

    def _showResult3D(self):
        if not self._resultCompareSegmentationNode:
            return False
        seg = self._resultCompareSegmentationNode
        seg.CreateClosedSurfaceRepresentation()
        self._setResultDisplaySettings()
        display = seg.GetDisplayNode()
        if display and hasattr(display, "SetPreferredDisplayRepresentationName3D"):
            try:
                closedName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
            except Exception:
                closedName = "Closed surface"
            display.SetPreferredDisplayRepresentationName3D(closedName)
        manager = slicer.app.layoutManager()
        layoutClass = getattr(slicer, "vtkMRMLLayoutNode", None)
        layoutValue = getattr(layoutClass, "SlicerLayoutFourUpView", None) if layoutClass else None
        if layoutValue is not None:
            manager.setLayout(layoutValue)
        threeDCount = getattr(manager, "threeDViewCount", 0)
        if callable(threeDCount):
            threeDCount = threeDCount()
        if threeDCount > 0:
            widget = manager.threeDWidget(0)
            view = widget.threeDView()
            view.show()
            if hasattr(view, "resetCamera"):
                view.resetCamera()
            if hasattr(view, "resetFocalPoint"):
                view.resetFocalPoint()
            if hasattr(view, "renderWindow"):
                view.renderWindow().Render()
        self._setDetails("3D 对比已显示\n人工标注：绿色\n模型预测：蓝色\n相机：已对准两者联合可见范围")
        return True

    def _exportCurrentExperiment(self):
        if not self._resultRunDir:
            self._setStatusMessage("只有已归档的真实实验可以导出；当前没有实验记录。", "warning")
            return False
        selected = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "选择实验导出目录", str(self.projectRoot / "workspace")
        )
        if not selected:
            return False
        destination = Path(selected) / "experiment_export"
        try:
            export_experiment_results(self._resultRunDir, destination=destination)
            screenshots = self._captureExperimentScreenshots(destination)
            self._setStatusMessage(f"实验结果已导出到：{destination}", "success")
            screenshotText = (
                "\n匿名可视化截图：" + ", ".join(str(path.name) for path in screenshots)
                if screenshots
                else "\n当前没有已加载的 GT / Prediction 对比，未生成可视化截图；先选择病例并查看对比后再导出。"
            )
            self._setDetails(f"实验导出完成\n目录：{destination}{screenshotText}")
            return True
        except Exception:
            self._setDetails("导出实验结果失败\n" + traceback.format_exc())
            self._setStatusMessage("导出没有完成，请查看高级信息。", "warning")
            return False

    def _captureExperimentScreenshots(self, destination):
        """Capture only anonymous platform views when a comparison is loaded."""

        if not self._resultCompareSegmentationNode or not hasattr(slicer.util, "saveScreenshot"):
            return []
        screenshotsDir = Path(destination) / "screenshots"
        screenshotsDir.mkdir(parents=True, exist_ok=True)
        saved = []
        manager = slicer.app.layoutManager()
        try:
            self._showResult2D()
            primaryName = next(iter(manager.sliceViewNames()), None)
            if primaryName:
                sliceWidget = manager.sliceWidget(primaryName)
                sliceView = sliceWidget.sliceView() if sliceWidget else None
                if sliceView:
                    path = screenshotsDir / "gt_prediction_slice.png"
                    slicer.util.saveScreenshot(str(path), sliceView)
                    if path.is_file():
                        saved.append(path)
        except Exception:
            pass
        try:
            self._showResult3D()
            threeDCount = getattr(manager, "threeDViewCount", 0)
            if callable(threeDCount):
                threeDCount = threeDCount()
            if int(threeDCount) > 0:
                view = manager.threeDWidget(0).threeDView()
                path = screenshotsDir / "gt_prediction_3d.png"
                slicer.util.saveScreenshot(str(path), view)
                if path.is_file():
                    saved.append(path)
        except Exception:
            pass
        return saved

    # ------------------------------------------------------------------
    # New-case prediction
    # ------------------------------------------------------------------
    def _refreshPredictionPage(self):
        if not hasattr(self, "predictionModelLabel"):
            return
        states = detect_fold_states(
            fold_results_directory(results_root=self.projectRoot / "workspace" / "nnUNet_results")
        )
        modelReady = all(state.completed for state in states)
        if modelReady:
            self.predictionModelLabel.setText("✓ 五折训练模型已准备好")
            self.predictionModelHintLabel.setText("将使用 fold 0–4 ensemble 预测新病例。")
        else:
            self.predictionModelLabel.setText("当前还没有完整的五折训练模型")
            done = len([state for state in states if state.completed])
            self.predictionModelHintLabel.setText(
                f"已完成 {done} / {len(states)} 组。请先完成正式训练，再使用自动分割。"
            )
        hasInput = bool(self._predictionInputPath and Path(self._predictionInputPath).exists())
        hasOutput = prediction_result_ready(self._predictionOutputPath)
        running = self._process is not None
        self.startPredictionButton.setEnabled(modelReady and hasInput and not running)
        self.selectPredictionButton.setEnabled(not running)
        self.viewPrediction2DButton.setEnabled(hasOutput and bool(self._predictionSegmentationNode))
        self.viewPrediction3DButton.setEnabled(hasOutput and bool(self._predictionSegmentationNode))
        self.exportPredictionButton.setEnabled(hasOutput)

    def _choosePredictionInput(self):
        selected = self._dialogPath(
            qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                "选择新的 MRI",
                str(self.projectRoot / "workspace" / "nifti"),
                "核磁文件 (*.nii.gz *.nii *.nrrd);;所有文件 (*)",
            )
        )
        if not selected:
            return False
        self._predictionInputPath = Path(selected).resolve()
        caseStem = self._stripImageSuffix(self._predictionInputPath)
        self._predictionOutputPath = (
            self.projectRoot / "workspace" / "predictions" / f"{caseStem}_condyle.nii.gz"
        ).resolve()
        self.predictionInputLabel.setText(str(self._predictionInputPath))
        self.predictionStatusLabel.setText("MRI 已选择，可以开始自动分割。")
        self._refreshPredictionPage()
        return True

    def _startPrediction(self):
        if not self._predictionInputPath or not Path(self._predictionInputPath).exists():
            self._setStatusMessage("请先选择新的 MRI。", "warning")
            return False
        self._refreshPredictionPage()
        if not self.startPredictionButton.isEnabled():
            self._setStatusMessage("需要完整五折模型和可用的输入 MRI 才能自动分割。", "warning")
            return False
        self.predictionLogWidget.clear()
        self.predictionStatusLabel.setText("正在调用五折 nnU-Net ensemble，请稍候…")
        command = prediction_command(
            self._predictionInputPath,
            self._predictionOutputPath,
            device="cuda",
            project_root=self.projectRoot,
            python_executable=project_python_executable(self.projectRoot),
        )
        return self._startExternalProcess("prediction", command)

    def _finishPrediction(self, exitCode):
        if exitCode != 0 or not prediction_result_ready(self._predictionOutputPath):
            self.predictionStatusLabel.setText("自动分割没有完成，未加载不存在的预测结果。")
            self._setStatusMessage("自动分割失败，请查看详细日志。", "warning")
            return
        try:
            self._loadPredictionResult()
            self.predictionStatusLabel.setText("自动分割完成 ✓ 已生成预测 mask，可以查看切片或 3D。")
            self._setStatusMessage("自动分割完成。", "success")
        except Exception:
            self._setDetails("显示预测结果失败\n" + traceback.format_exc())
            self.predictionStatusLabel.setText("预测文件已生成，但 3D Slicer 没有成功显示。")
            self._setStatusMessage("预测文件存在，但显示没有完成。", "warning")
        self._refreshPredictionPage()

    def _loadPredictionResult(self):
        if not self._predictionInputPath or not self._predictionOutputPath:
            raise RuntimeError("prediction paths are empty")
        self._removeNodeSafely(getattr(self, "_predictionImageNode", None))
        self._removeNodeSafely(self._predictionLabelNode)
        self._removeNodeSafely(self._predictionSegmentationNode)
        loaded = slicer.util.loadVolume(str(self._predictionInputPath), returnNode=True)
        self._predictionImageNode = loaded[1] if isinstance(loaded, tuple) else loaded
        labelLoaded = slicer.util.loadLabelVolume(str(self._predictionOutputPath), returnNode=True)
        labelNode = labelLoaded[1] if isinstance(labelLoaded, tuple) else labelLoaded
        if not self._predictionImageNode or not labelNode:
            raise RuntimeError("Slicer could not load prediction pair")
        seg = self._segmentationFromLabel(
            self._predictionImageNode,
            labelNode,
            "TMJ_Prediction",
            "自动预测",
            (0.1, 0.4, 0.95),
        )
        self._predictionSegmentationNode = seg
        self._predictionLabelNode = None
        self._showVolumeInSliceViews(self._predictionImageNode)
        self._showPrediction2D()

    def _segmentationFromLabel(self, imageNode, labelNode, nodeName, segmentName, color):
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", nodeName)
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(imageNode)
        imported = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, seg)
        self._removeNodeSafely(labelNode)
        if imported is False or seg.GetSegmentation().GetNumberOfSegments() < 1:
            self._removeNodeSafely(seg)
            raise RuntimeError("prediction label is empty or could not be imported")
        segmentId = seg.GetSegmentation().GetNthSegmentID(0)
        segment = seg.GetSegmentation().GetSegment(segmentId)
        segment.SetName(segmentName)
        segment.SetColor(*color)
        display = seg.GetDisplayNode()
        if display:
            display.SetVisibility(True)
            if hasattr(display, "SetOpacity"):
                display.SetOpacity(0.55)
            if hasattr(display, "SetOpacity3D"):
                display.SetOpacity3D(0.8)
            if hasattr(display, "SetSegmentVisibility"):
                display.SetSegmentVisibility(segmentId, True)
            if hasattr(display, "SetSegmentVisibility3D"):
                display.SetSegmentVisibility3D(segmentId, True)
        return seg

    def _showPrediction2D(self):
        if not self._predictionSegmentationNode or not getattr(self, "_predictionImageNode", None):
            return False
        self._setCheckLayout()
        self._showVolumeInSliceViews(self._predictionImageNode)
        return True

    def _showPrediction3D(self):
        seg = self._predictionSegmentationNode
        if not seg:
            return False
        seg.CreateClosedSurfaceRepresentation()
        self._setCheckLayout()
        display = seg.GetDisplayNode()
        if display and hasattr(display, "SetVisibility3D"):
            display.SetVisibility3D(True)
        manager = slicer.app.layoutManager()
        threeDCount = getattr(manager, "threeDViewCount", 0)
        if callable(threeDCount):
            threeDCount = threeDCount()
        self._showVolumeInSliceViews(self._predictionImageNode)
        if int(threeDCount) > 0:
            view = manager.threeDWidget(0).threeDView()
            view.show()
            if hasattr(view, "resetCamera"):
                view.resetCamera()
            if hasattr(view, "resetFocalPoint"):
                view.resetFocalPoint()
            if hasattr(view, "renderWindow"):
                view.renderWindow().Render()
        self._setStatusMessage("预测髁突 3D 已显示，相机已自动对准模型。", "success")
        return True

    def _onPredictionOpacityChanged(self, value):
        self.predictionOpacityLabel.setText(f"{int(value)}%")
        display = self._predictionSegmentationNode.GetDisplayNode() if self._predictionSegmentationNode else None
        if display:
            opacity = float(value) / 100.0
            try:
                display.SetOpacity(opacity)
                if hasattr(display, "SetOpacity3D"):
                    display.SetOpacity3D(min(0.95, opacity + 0.2))
            except Exception:
                pass

    def _exportPrediction(self):
        if not prediction_result_ready(self._predictionOutputPath):
            self._setStatusMessage("当前没有可以导出的预测 mask。", "warning")
            return False
        selected = self._dialogPath(
            qt.QFileDialog.getSaveFileName(
                slicer.util.mainWindow(),
                "导出预测 mask",
                str(self._predictionOutputPath),
                "NIfTI mask (*.nii.gz);;所有文件 (*)",
            )
        )
        if not selected:
            return False
        destination = Path(selected)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._predictionOutputPath, destination)
        self._setStatusMessage(f"预测 mask 已导出：{destination}", "success")
        return True

    def _showTrainingLogDialog(self):
        textValue = self._processOutput
        if self._currentRunDir:
            logPath = self._currentRunDir / "logs" / "training.log"
            if logPath.exists():
                try:
                    textValue = logPath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("详细日志")
        dialog.setMinimumSize(820, 520)
        dialogLayout = qt.QVBoxLayout(dialog)
        log = qt.QPlainTextEdit()
        log.setReadOnly(True)
        log.setPlainText(textValue or "暂无日志。")
        dialogLayout.addWidget(log)
        close = self._secondaryButton("关闭")
        close.clicked.connect(lambda checked=False: dialog.accept())
        dialogLayout.addWidget(close, 0, qt.Qt.AlignRight)
        self._execDialog(dialog)

    def _showSettingsAdvanced(self):
        self.advancedToggle.setChecked(True)
        self._showDetailsDialog()

    def _currentSlicerExecutable(self):
        try:
            value = getattr(slicer.app, "applicationFilePath")
            value = value() if callable(value) else value
            candidate = Path(str(value)).resolve()
            if candidate.name.lower() == "slicer.exe" and candidate.is_file():
                return candidate
            fallback = candidate.parent.parent / "Slicer.exe"
            if fallback.is_file():
                return fallback.resolve()
        except Exception:
            pass
        return None

    def _changeSlicerPath(self):
        selected = self._dialogPath(
            qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                "选择 Slicer.exe",
                str(Path.home()),
                "Slicer.exe (Slicer.exe);;可执行文件 (*.exe)",
            )
        )
        if not selected:
            return False
        path = Path(selected).resolve()
        if path.name.lower() != "slicer.exe" or not path.is_file():
            self._setStatusMessage("请选择文件名为 Slicer.exe 的程序。", "warning")
            return False
        try:
            write_slicer_config(path, project_root=self.projectRoot)
        except Exception as exc:
            self._setDetails(f"保存 Slicer 路径失败\n{type(exc).__name__}: {exc}")
            self._setStatusMessage("Slicer 路径没有保存，请重试。", "warning")
            return False
        self._setStatusMessage("Slicer 路径已保存；下次启动实验平台会使用这个版本。", "success")
        self._refreshSettingsPage()
        return True

    def _redetectSlicer(self):
        try:
            candidates = discover_slicer_candidates(project_root=self.projectRoot)
        except Exception as exc:
            self._setDetails(f"Slicer 检测失败\n{traceback.format_exc()}")
            self._setStatusMessage(f"Slicer 检测失败：{exc}", "warning")
            return False
        if not candidates:
            self._setStatusMessage(
                "没有找到 3D Slicer，请点击“更换”选择 Slicer.exe。", "warning"
            )
            self._refreshSettingsPage()
            return False
        chosen = candidates[0]
        if len(candidates) > 1:
            labels = [f"{item.path}  [{item.source}]" for item in candidates]
            selected, accepted = qt.QInputDialog.getItem(
                slicer.util.mainWindow(),
                "选择 3D Slicer",
                "找到多个版本，请选择一个：",
                labels,
                0,
                False,
            )
            if not accepted:
                return False
            chosen = candidates[labels.index(str(selected))]
        try:
            write_slicer_config(chosen.path, project_root=self.projectRoot)
        except Exception as exc:
            self._setDetails(f"保存 Slicer 路径失败\n{traceback.format_exc()}")
            self._setStatusMessage(f"Slicer 路径没有保存：{exc}", "warning")
            return False
        self._setStatusMessage("Slicer 检测完成。", "success")
        self._refreshSettingsPage()
        return True

    def _openProjectPath(self, path):
        destination = Path(path)
        if not destination.exists():
            self._setStatusMessage("这个目录目前还不存在。完成对应实验后再打开。", "info")
            return False
        try:
            opened = qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(destination)))
            if not opened:
                self._setStatusMessage("系统没有打开这个目录。", "warning")
            return bool(opened)
        except Exception:
            self._setDetails("打开目录失败\n" + traceback.format_exc())
            self._setStatusMessage("目录没有打开，请查看技术信息。", "warning")
            return False

    def _openLatestLog(self):
        candidates = sorted(
            (
                path
                for path in (self.projectRoot / "workspace").rglob("*.log")
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return self._openProjectPath(candidates[0])
        self._setStatusMessage("当前还没有后台任务日志。", "info")
        self._showDetailsDialog()
        return False

    def _refreshSettingsPage(self):
        if not hasattr(self, "settingsSlicerStatusLabel"):
            return
        configured = configured_slicer_path(project_root=self.projectRoot)
        current = self._currentSlicerExecutable()
        selected = configured or current
        self._slicerCandidates = discover_slicer_candidates(project_root=self.projectRoot)
        if selected:
            self.settingsSlicerStatusLabel.setText("✓ 已找到")
            self.settingsSlicerVersionLabel.setText(self._slicerVersion())
            self.settingsSlicerPathLabel.setText("已找到（完整路径已隐藏）")
        else:
            self.settingsSlicerStatusLabel.setText("未找到")
            self.settingsSlicerVersionLabel.setText("—")
            self.settingsSlicerPathLabel.setText("尚未配置，请点击“更换”")
        report = self._environmentReport
        if report:
            display = environment_display(report)
            self.settingsTrainingLabel.setText(
                "nnU-Net　{0} {1}".format(
                    display["nnunet"],
                    "已安装" if display["nnunet"] == "✓" else "未安装",
                )
            )
            self.settingsGpuLabel.setText(display["gpu_message"])
        else:
            self.settingsTrainingLabel.setText("nnU-Net　尚未检测")
            self.settingsGpuLabel.setText("尚未检测；进入模型训练页会自动检查。")
        detailLines = [
            "平台设置",
            f"Slicer 配置文件：{slicer_config_path(self.projectRoot)}",
            f"当前 Slicer：{selected or '未找到'}",
            f"项目 Python：{project_python_executable(self.projectRoot)}",
            f"训练数据目录：{self.projectRoot / 'workspace' / 'nnUNet_raw' / 'Dataset501_CondyleMRI'}",
            f"工作区：{self.projectRoot / 'workspace'}",
        ]
        if report:
            detailLines.append("环境报告：\n" + json.dumps(report, ensure_ascii=False, indent=2))
        self._setDetails("\n".join(str(line) for line in detailLines))

    def _slicerVersion(self):
        try:
            value = getattr(slicer.app, "applicationVersion")
            value = value() if callable(value) else value
            if value:
                return str(value)
        except Exception:
            pass
        return "已安装"

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
        dialog.setMinimumSize(520, 460)
        dialog.setStyleSheet(self._styleSheet())
        layout = qt.QVBoxLayout(dialog)
        title = qt.QLabel("怎么做一次完整实验？")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        text = qt.QLabel(
            "完成一次实验只需要下面几步：\n\n"
            "1. 导入核磁病例\n"
            "2. 把下颌髁突标出来并保存\n"
            "3. 准备至少几例已确认病例\n"
            "4. 在“训练数据”中检查并准备训练数据\n"
            "5. 在“模型训练”中开始 5 组交叉验证\n"
            "6. 等待 5 组实验完成\n"
            "7. 在“实验结果”中查看 Dice、IoU 和 HD95\n"
            "8. 选择病例查看人工标注与自动预测的 2D / 3D 对比\n"
            "9. 在“自动分割”中用新的 MRI 测试模型\n\n"
            "系统只显示真实训练和评价结果；没有足够病例或兼容 CUDA 显卡时，"
            "会明确提示原因，不会生成假指标。"
        )
        text.setObjectName("mutedLabel")
        text.setWordWrap(True)
        layout.addWidget(text)
        closeButton = self._primaryButton("知道了")
        closeButton.clicked.connect(lambda checked=False: dialog.accept())
        layout.addWidget(closeButton)
        self._execDialog(dialog)

    def _showFirstRunWizardIfNeeded(self):
        """Orient a first-time user before showing the full workbench."""

        if self._firstRunWizardDialog is not None:
            try:
                if self._firstRunWizardDialog.isVisible():
                    return
            except Exception:
                pass
        try:
            completed = slicer.app.settings().value(
                "TMJCondyleAnnotator/firstRunWizardCompleted", ""
            )
        except Exception:
            completed = ""
        if not should_show_first_run_wizard(completed):
            return

        dialog = qt.QDialog(slicer.util.mainWindow())
        self._firstRunWizardDialog = dialog
        dialog.setWindowTitle("首次使用引导")
        dialog.setMinimumSize(560, 430)
        dialog.setStyleSheet(self._styleSheet())
        layout = qt.QVBoxLayout(dialog)
        stack = qt.QStackedWidget()
        layout.addWidget(stack, 1)

        def addPage(eyebrow, title, body):
            page = qt.QWidget()
            pageLayout = qt.QVBoxLayout(page)
            pageLayout.setContentsMargins(24, 24, 24, 18)
            eyebrowLabel = qt.QLabel(eyebrow)
            eyebrowLabel.setObjectName("pageEyebrow")
            pageLayout.addWidget(eyebrowLabel)
            titleLabel = qt.QLabel(title)
            titleLabel.setObjectName("pageTitle")
            pageLayout.addWidget(titleLabel)
            bodyLabel = qt.QLabel(body)
            bodyLabel.setObjectName("homePurpose")
            bodyLabel.setWordWrap(True)
            pageLayout.addWidget(bodyLabel)
            pageLayout.addStretch(1)
            stack.addWidget(page)

        addPage(
            "欢迎",
            "这是做什么的？",
            "这个软件可以先把 MRI 中的下颌髁突标出来，\n"
            "再用这些标注训练电脑。\n\n"
            "训练完成以后，\n"
            "新的 MRI 可以自动分割出下颌髁突，\n"
            "并显示三维模型。",
        )
        addPage(
            "实验流程",
            "整个实验只需要几步",
            "1. 导入病例\n"
            "2. 标注下颌髁突\n"
            "3. 准备训练数据\n"
            "4. 训练模型\n"
            "5. 查看实验结果\n"
            "6. 用新的 MRI 自动分割",
        )
        addPage(
            "开始之前",
            "第一步先做什么？",
            "先准备并标注一些 MRI。\n\n"
            "建议先标 5 例检查流程，\n"
            "确认没有问题后继续增加病例。",
        )

        footer = qt.QHBoxLayout()
        neverAgain = qt.QCheckBox("下次不再显示")
        footer.addWidget(neverAgain)
        footer.addStretch(1)
        backButton = self._secondaryButton("上一步")
        backButton.setVisible(False)
        footer.addWidget(backButton)
        nextButton = self._primaryButton("下一步")
        footer.addWidget(nextButton)
        startButton = self._primaryButton("开始")
        startButton.setVisible(False)
        footer.addWidget(startButton)
        layout.addLayout(footer)

        def stackCount():
            value = stack.count
            return int(value() if callable(value) else value)

        def stackIndex():
            value = stack.currentIndex
            return int(value() if callable(value) else value)

        def updateButtons(index):
            backButton.setVisible(index > 0)
            nextButton.setVisible(index < stackCount() - 1)
            startButton.setVisible(index == stackCount() - 1)

        def goNext(checked=False):
            index = min(stackIndex() + 1, stackCount() - 1)
            stack.setCurrentIndex(index)
            updateButtons(index)

        def goBack(checked=False):
            index = max(stackIndex() - 1, 0)
            stack.setCurrentIndex(index)
            updateButtons(index)

        def begin(checked=False):
            if neverAgain.isChecked():
                try:
                    slicer.app.settings().setValue(
                        "TMJCondyleAnnotator/firstRunWizardCompleted", "true"
                    )
                except Exception:
                    pass
            dialog.accept()
            self._showHome()

        nextButton.clicked.connect(goNext)
        backButton.clicked.connect(goBack)
        startButton.clicked.connect(begin)
        dialog.finished.connect(lambda result=0: setattr(self, "_firstRunWizardDialog", None))
        updateButtons(0)
        self._execDialog(dialog)

    def _showDemoDialog(self):
        """Show a read-only phantom workflow so beginners can learn the UI."""

        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("软件演示")
        dialog.setMinimumSize(560, 470)
        dialog.setStyleSheet(self._styleSheet())
        layout = qt.QVBoxLayout(dialog)
        title = qt.QLabel("查看软件演示")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        notice = qt.QLabel(
            "这是只读的 synthetic / phantom 流程演示，不读取、不修改真实患者数据，"
            "也不会把演示内容加入正式训练数据集。"
        )
        notice.setObjectName("resultMessage")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        steps = qt.QListWidget()
        steps.addItems(
            [
                "✓ 1　导入病例（演示）",
                "✓ 2　标注下颌髁突（演示）",
                "○ 3　确认标注后准备训练数据",
                "○ 4　训练 5 折模型",
                "○ 5　查看 Dice / IoU / HD95",
                "○ 6　导入新的 MRI 自动分割并查看 3D",
            ]
        )
        steps.setEnabled(False)
        layout.addWidget(steps, 1)
        explanation = qt.QLabel(
            "正式操作时，请按首页的“下一步”按钮进行。只有“已确认”的人工标注才会进入训练。"
        )
        explanation.setObjectName("mutedLabel")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        closeButton = self._primaryButton("知道了")
        closeButton.clicked.connect(lambda checked=False: dialog.accept())
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
                    "menubar",
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
                try:
                    menuBar = mainWindow.menuBar()
                    menuId = id(menuBar) if menuBar is not None else None
                    if menuBar is not None and menuId not in existing:
                        self._simpleModeTargets.append(
                            (menuBar, bool(menuBar.isVisible()))
                        )
                        existing.add(menuId)
                        menuBar.setVisible(False)
                    elif menuBar is not None:
                        menuBar.setVisible(False)
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
                "显示完整 Slicer" if self._simpleMode else "开启简洁模式"
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
        self._homeVisible = False
        self._mainPage = "cases"
        self.pageStack.setCurrentIndex(index + 1)
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

    def _showHome(self):
        if self._dirty and not self._confirmUnsaved():
            return False
        self._homeVisible = True
        self._mainPage = "home"
        self.pageStack.setCurrentIndex(PAGE_HOME)
        self._restoreWorkflowLayout()
        self._setStatusMessage("欢迎使用。首页会根据当前状态告诉你下一步做什么。", "info")
        self._updatePageChrome()
        self._syncUi()
        return True

    def _goHome(self, checked=False):
        return self._showHome()

    def _runHomeNextAction(self, checked=False):
        inventory, counts = self._inventoryAndCounts()
        action = self._homeNextAction(counts)
        if action.key == "cases":
            self._showMainPage("cases")
            if counts.get("total", 0) and action.button != "开始导入病例":
                self._continueLastAnnotation()
            return True
        if action.key == "dataset":
            return self._showMainPage("dataset")
        if action.key == "training":
            return self._showMainPage("training")
        if action.key == "results":
            return self._showMainPage("results")
        if action.key == "prediction":
            return self._showMainPage("prediction")
        return False

    def _startNewAnnotationFromHome(self, checked=False):
        self._showPage(0)
        self._setStatusMessage("请选择一份核磁文件，或导入病例文件夹。", "info")

    def _continueLastAnnotation(self, checked=False):
        if self.volumeNode:
            self._startAnnotation()
            return
        self._refreshCaseFiles()
        if not self._caseFiles:
            self._showPage(0)
            self._setStatusMessage("还没有可继续的病例，请先选择核磁文件。", "info")
            return
        target = 0
        for index in range(len(self._caseFiles)):
            if self._caseStatusForIndex(index) != "已确认":
                target = index
                break
        if self._loadCaseAtIndex(target):
            self._startAnnotation()

    def _showProgressDialog(self, checked=False, onlyCompleted=False):
        self._refreshCaseFiles()
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("标注进度")
        dialog.setMinimumSize(500, 420)
        dialog.setStyleSheet(self._styleSheet())
        layout = qt.QVBoxLayout(dialog)
        title = qt.QLabel("病例进度")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        summary = qt.QLabel()
        summary.setObjectName("mutedLabel")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        rows = qt.QListWidget()
        rows.setObjectName("progressList")
        layout.addWidget(rows, 1)
        rowToCase = []
        completed = 0
        for index, path in enumerate(self._caseFiles):
            status = self._caseStatusForIndex(index)
            if status in {"已标注", "已确认"}:
                completed += 1
            if onlyCompleted and status not in {"已标注", "已确认"}:
                continue
            item = qt.QListWidgetItem(
                f"{self._caseIdForIndex(path, index)}    {self._statusSymbol(status)} {status}"
            )
            rows.addItem(item)
            rowToCase.append(index)
        summary.setText(
            f"{len(self._caseFiles)} 个病例　　已完成：{completed}　　未完成：{len(self._caseFiles) - completed}"
        )
        if not rowToCase:
            rows.addItem("当前没有符合条件的病例。")

        buttonRow = qt.QHBoxLayout()
        continueButton = self._primaryButton("继续标注")
        closeButton = self._secondaryButton("关闭")
        buttonRow.addStretch(1)
        buttonRow.addWidget(closeButton)
        buttonRow.addWidget(continueButton)
        layout.addLayout(buttonRow)

        def openSelected(*args):
            row = rows.currentRow
            if row < 0 or row >= len(rowToCase):
                return
            index = rowToCase[row]
            if self._loadCaseAtIndex(index):
                dialog.accept()
                self._startAnnotation()

        continueButton.clicked.connect(openSelected)
        rows.itemDoubleClicked.connect(lambda *args: openSelected())
        closeButton.clicked.connect(lambda checked=False: dialog.accept())
        self._execDialog(dialog)

    @staticmethod
    def _statusSymbol(status):
        return {
            "已确认": "✓",
            "已标注": "●",
            "标注中": "●",
            "未验证": "!",
            "未标注": "○",
            "未开始": "○",
        }.get(status, "○")

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
        if self._foregroundVoxelCount() <= 0:
            self._annotationHasData = False
            self._setStatusMessage("还没有标出髁突，暂时无法生成三维效果。", "warning")
            self._setResultMessage(
                self.annotationMessageLabel,
                "还没有标出髁突，先用画笔标出一部分再检查 3D。",
                "warning",
            )
            self._syncUi()
            return False
        self._setPageForCheck()
        return True

    def _setPageForCheck(self):
        self._currentPage = 2
        self._homeVisible = False
        self._mainPage = "cases"
        self.pageStack.setCurrentIndex(PAGE_CASE_CHECK)
        self._updatePageChrome()
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

    def _show2DCheckView(self, checked=False):
        if not self.segmentationNode:
            self._setStatusMessage("请先开始标注。", "warning")
            return False
        self._setAnnotationLayout()
        self._setStatusMessage("已切换到切片视图，可以逐层复核标注。", "info")
        return True

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
        resolvedPath = Path(path).resolve() if path else None
        if resolvedPath in self._batchCaseIds:
            return self._batchCaseIds[resolvedPath]
        candidate = self._safeCaseId(path, fallback="")
        return candidate or f"case_{index + 1:03d}"

    @staticmethod
    def _supportedImageFiles(folder):
        folder = Path(folder)
        paths = []
        for pattern in ("*.nii.gz", "*.nii", "*.nrrd"):
            paths.extend(folder.rglob(pattern))
        return sorted(
            {path.resolve() for path in paths if path.is_file()},
            key=lambda path: (path.name.lower(), str(path).lower()),
        )

    def _setBatchFiles(self, paths, autoNumber=False):
        paths = [Path(path).resolve() for path in paths]
        self._batchCaseFiles = paths
        self._batchCaseIds = (
            {path: f"case_{index + 1:03d}" for index, path in enumerate(paths)}
            if autoNumber
            else {}
        )
        self._refreshCaseFiles()

    def _refreshCaseFiles(self):
        if self._batchCaseFiles is not None:
            paths = list(self._batchCaseFiles)
        else:
            paths = self._supportedImageFiles(self.niftiDir)
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
        self._refreshCaseListWidget()

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
                "选择核磁文件",
                str(self.niftiDir),
                "核磁文件 (*.nii.gz *.nii *.nrrd);;所有文件 (*)",
            )
        )
        if not fileName:
            return
        if not self._confirmUnsaved():
            return
        selectedPath = Path(fileName).resolve()
        if selectedPath.parent == self.niftiDir.resolve():
            self._batchCaseFiles = None
            self._batchCaseIds = {}
        else:
            self._setBatchFiles([selectedPath], autoNumber=True)
        self._loadVolumePath(selectedPath, confirm=False)

    def _chooseAndLoadFolder(self):
        folder = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "选择病例文件夹",
            str(self.niftiDir),
        )
        if not folder:
            return
        if not self._confirmUnsaved():
            return
        paths = self._supportedImageFiles(folder)
        if not paths:
            self._setStatusMessage(
                "这个文件夹里没有找到 .nii.gz、.nii 或 .nrrd 文件。", "warning"
            )
            return
        self._setBatchFiles(paths, autoNumber=True)
        self._loadVolumePath(paths[0], confirm=False)

    def _loadVolumePath(self, fileName, confirm=True):
        if confirm and not self._confirmUnsaved():
            return False
        try:
            loaded = slicer.util.loadVolume(str(fileName), returnNode=True)
            node = loaded[1] if isinstance(loaded, tuple) else loaded
        except TypeError:
            try:
                node = slicer.util.loadVolume(str(fileName))
            except Exception as exc:
                self._handleLoadFailure(fileName, exc)
                return False
        except Exception as exc:
            self._handleLoadFailure(fileName, exc)
            return False
        if not node:
            self._handleLoadFailure(fileName, "没有返回可用的 MRI")
            return False
        self._ownedVolumeIds.add(node.GetID())
        self._setVolumeNode(node, Path(fileName).resolve())
        return True

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
        self._currentAnnotationStatus = self._manifestStatusForCase(self._currentCaseId)
        if not self._currentAnnotationStatus:
            self._currentAnnotationStatus = "ANNOTATED" if self._saved else "NEW"
        if self._saved and self._currentAnnotationStatus in {"NEW", "ANNOTATING"}:
            self._currentAnnotationStatus = "UNVERIFIED"
        self._currentPage = 0
        self._homeVisible = False
        self._threeDVisible = False
        self._surfacePointCount = 0
        self._surfaceCellCount = 0
        self._mainPage = "cases"
        self.pageStack.setCurrentIndex(PAGE_CASE_IMPORT)
        self._updatePageChrome()
        self.saveSuccessLabel.setVisible(False)
        self.saveResultLabel.clear()
        self.saveNextGuidanceLabel.clear()
        self.nextCaseAfterSaveButton.setText("继续下一个病例")
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
        layerCount = int(node.GetImageData().GetDimensions()[self._primaryAxis])
        self._setResultMessage(
            self.importResultLabel,
            f"✓ 已导入\n病例：{self._currentCaseId}\n图像层数：{layerCount} 层",
            "success",
        )
        self._setStatusMessage("病例已导入，可以开始标注髁突。", "success")
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

    def _manifestStatusForCase(self, caseId):
        try:
            rows = read_manifest(self.manifestPath)
        except Exception:
            return ""
        row = next((item for item in rows if item.get("case_id") == caseId), None)
        if not row:
            return ""
        status = str(row.get("annotation_status") or "").strip().upper()
        return status if status in {"NEW", "ANNOTATING", "ANNOTATED", "VERIFIED", "UNVERIFIED"} else "UNVERIFIED"

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
        return self._loadVolumePath(path, confirm=False)

    def _loadCaseAtIndex(self, index):
        self._refreshCaseFiles()
        if index < 0 or index >= len(self._caseFiles):
            return False
        if self.volumeNode and self._currentCaseIndex == index:
            return True
        path = self._caseFiles[index]
        return self._loadVolumePath(path, confirm=True)

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
                "下颌髁突",
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
        self._threeDVisible = False
        self._surfacePointCount = 0
        self._surfaceCellCount = 0
        self._syncUi()
        return True

    def _foregroundVoxelCount(self):
        if not self.segmentationNode or not self.segmentId:
            return 0
        try:
            array = slicer.util.arrayFromSegmentInternalBinaryLabelmap(
                self.segmentationNode, self.segmentId
            )
            if array is not None:
                return int(np.count_nonzero(np.asarray(array)))
        except Exception:
            pass
        try:
            segment = self.segmentationNode.GetSegmentation().GetSegment(
                self.segmentId
            )
            binary = segment.GetRepresentation("Binary labelmap")
            if binary and binary.GetImageData():
                return int(binary.GetImageData().GetScalarRange()[1] > 0)
        except Exception:
            pass
        return 0

    def _caseStatusForIndex(self, index):
        if index < 0 or index >= len(self._caseFiles):
            return "未开始"
        path = self._caseFiles[index]
        if self._currentCasePath and Path(path).resolve() == self._currentCasePath:
            if self._saved and not self._dirty:
                return {
                    "VERIFIED": "已确认",
                    "UNVERIFIED": "未验证",
                    "ANNOTATED": "已标注",
                }.get(
                    self._currentAnnotationStatus,
                    "未验证" if self._annotationHasData else "未标注",
                )
            if self._foregroundVoxelCount() > 0:
                return "标注中"
        caseId = self._caseIdForIndex(path, index)
        status = self._manifestStatusForCase(caseId)
        if status == "VERIFIED" and (self.labelsDir / f"{caseId}.nii.gz").exists():
            return "已确认"
        if status == "ANNOTATED" and (self.labelsDir / f"{caseId}.nii.gz").exists():
            return "已标注"
        if status == "UNVERIFIED" or (self.labelsDir / f"{caseId}.nii.gz").exists():
            return "未验证"
        if status == "ANNOTATING":
            return "标注中"
        return "未标注"

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
            if self.segmentId and hasattr(display, "SetSegmentVisibility3D"):
                display.SetSegmentVisibility3D(self.segmentId, True)
            if self.segmentId and hasattr(display, "SetSegmentVisibility2DFill"):
                display.SetSegmentVisibility2DFill(self.segmentId, True)
            if self.segmentId and hasattr(display, "SetSegmentVisibility2DOutline"):
                display.SetSegmentVisibility2DOutline(self.segmentId, True)
        except Exception:
            pass

    def _loadExistingLabelIfPresent(self):
        path = self._outputPath()
        if not path.exists() or not self.segmentationNode:
            return
        if not self._currentAnnotationStatus or self._currentAnnotationStatus == "NEW":
            # A mask without a trustworthy manifest record is never treated as
            # a verified training label.
            self._currentAnnotationStatus = "UNVERIFIED"
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
                segment.SetName("下颌髁突")
                segment.SetColor(0.16, 0.72, 0.78)
            self._annotationHasData = self._foregroundVoxelCount() > 0
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
        self._currentAnnotationStatus = "ANNOTATED"
        self._annotationHasData = self._foregroundVoxelCount() > 0
        self._qcStatus = "未检查"
        self._threeDVisible = False
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
        foreground = self._foregroundVoxelCount()
        if foreground <= 0:
            self._threeDVisible = False
            self._surfacePointCount = 0
            self._surfaceCellCount = 0
            self._setResultMessage(
                self.qcResultLabel,
                "还没有标出髁突，暂时无法生成三维效果。\n请返回继续标注。",
                "warning",
            )
            if not silent:
                self._setStatusMessage("还没有标出髁突，暂时无法生成三维效果。", "warning")
            self._syncUi()
            return False
        try:
            segmentation = self.segmentationNode.GetSegmentation()
            if segmentation.GetNumberOfSegments() != 1 or not self.segmentId:
                raise RuntimeError("expected exactly one condyle segment")

            # Keep the official segmentation conversion pipeline.  Recreating
            # the closed surface here also refreshes it after a paint/erase
            # change instead of relying on a stale cached representation.
            self.segmentationNode.CreateClosedSurfaceRepresentation()
            slicer.app.processEvents()
            self._applySegmentationDisplaySettings()

            segment = segmentation.GetSegment(self.segmentId)
            closedName = "Closed surface"
            try:
                closedName = (
                    slicer.vtkSegmentationConverter
                    .GetSegmentationClosedSurfaceRepresentationName()
                )
            except Exception:
                pass
            surface = segment.GetRepresentation(closedName)
            if surface is None and closedName != "Closed surface":
                surface = segment.GetRepresentation("Closed surface")
            if surface is None:
                raise RuntimeError("closed surface representation was not created")
            pointCount = (
                int(surface.GetNumberOfPoints())
                if hasattr(surface, "GetNumberOfPoints")
                else 0
            )
            cellCount = (
                int(surface.GetNumberOfCells())
                if hasattr(surface, "GetNumberOfCells")
                else 0
            )
            if pointCount <= 0 or cellCount <= 0:
                raise RuntimeError(
                    f"closed surface is empty (points={pointCount}, cells={cellCount})"
                )
            surfaceBounds = [0.0] * 6
            surface.GetBounds(surfaceBounds)
            if not all(np.isfinite(value) for value in surfaceBounds):
                raise RuntimeError("closed surface bounds are invalid")
            surfaceSize = max(
                surfaceBounds[1] - surfaceBounds[0],
                surfaceBounds[3] - surfaceBounds[2],
                surfaceBounds[5] - surfaceBounds[4],
            )
            if surfaceSize <= 0:
                raise RuntimeError("closed surface bounds are empty")
            display = self.segmentationNode.GetDisplayNode()
            if display and hasattr(display, "SetPreferredDisplayRepresentationName3D"):
                display.SetPreferredDisplayRepresentationName3D(closedName)
            if display and hasattr(display, "SetSegmentOpacity3D"):
                display.SetSegmentOpacity3D(self.segmentId, 0.9)
            if display and hasattr(display, "SetSegmentVisibility3D"):
                display.SetSegmentVisibility3D(self.segmentId, True)

            # Change to the four-up layout so the 3D widget is definitely
            # present, then fit the camera to the visible segmentation actor.
            self._setCheckLayout()
            manager = slicer.app.layoutManager()
            threeDCount = getattr(manager, "threeDViewCount", 0)
            if callable(threeDCount):
                threeDCount = threeDCount()
            if int(threeDCount) <= 0:
                raise RuntimeError("the current Slicer layout has no 3D view")
            widget = manager.threeDWidget(0)
            view = widget.threeDView()
            view.show()
            if hasattr(view, "resetCamera"):
                view.resetCamera()
            if hasattr(view, "resetFocalPoint"):
                view.resetFocalPoint()
            # Slicer's generic reset can use the full reference-image bounds,
            # which makes a small condyle look like a missing surface.  Keep
            # the official 3D view/camera and retarget it to the actual
            # closed-surface bounds before the final render.
            cameraNode = view.cameraNode() if hasattr(view, "cameraNode") else None
            if cameraNode:
                center = [
                    (surfaceBounds[0] + surfaceBounds[1]) * 0.5,
                    (surfaceBounds[2] + surfaceBounds[3]) * 0.5,
                    (surfaceBounds[4] + surfaceBounds[5]) * 0.5,
                ]
                try:
                    oldPosition = [float(value) for value in cameraNode.GetPosition()]
                    oldFocal = [float(value) for value in cameraNode.GetFocalPoint()]
                    direction = [oldPosition[i] - oldFocal[i] for i in range(3)]
                except Exception:
                    direction = [0.0, 0.0, 1.0]
                directionLength = float(np.linalg.norm(direction))
                if directionLength <= 1e-6:
                    direction = [0.0, 0.0, 1.0]
                    directionLength = 1.0
                # The default camera may be hundreds of millimetres away when
                # the layout has no visible actor yet.  Reusing that distance
                # is precisely what makes a small surface look absent in a
                # perspective view, so derive the distance only from the
                # actual surface size.
                distance = max(surfaceSize * 3.0, 1.0)
                normalized = [value / directionLength for value in direction]
                position = [
                    center[i] + normalized[i] * distance for i in range(3)
                ]
                cameraNode.SetFocalPoint(center)
                cameraNode.SetPosition(position)
                cameraNode.SetParallelScale(max(surfaceSize * 0.9, 0.5))
                cameraNode.Modified()
                cameraNode.GetCamera().SetClippingRange(
                    0.1, max(distance + surfaceSize * 5.0, 10.0)
                )
            if hasattr(view, "renderWindow"):
                view.renderWindow().Render()
            if cameraNode:
                # The first render attaches the segmentation actor.  Apply
                # the near/far range once more so Slicer's automatic clipping
                # calculation cannot leave the actor behind the near plane.
                cameraNode.GetCamera().SetClippingRange(
                    0.1, max(distance + surfaceSize * 5.0, 10.0)
                )
                cameraNode.Modified()
                if hasattr(view, "renderWindow"):
                    view.renderWindow().Render()
            slicer.app.processEvents()

            self._threeDVisible = True
            self._surfacePointCount = pointCount
            self._surfaceCellCount = cellCount
            self._setDetails(
                f"三维显示完成\n病例：{self._currentCaseId}\n"
                f"前景体素数：{foreground}\n"
                f"Closed surface 点数：{pointCount}\n"
                f"Closed surface 面数：{cellCount}\n"
                f"surface bounds：{[round(value, 3) for value in surfaceBounds]}\n"
                "3D visibility：已打开\n相机：已自动对准髁突范围"
            )
            self._setResultMessage(
                self.qcResultLabel,
                "✓ 三维髁突已显示\n请旋转查看形状，再点击“检查标注”。",
                "success",
            )
            self._setStatusMessage("三维髁突已显示，请旋转检查轮廓。", "success")
            self._syncUi()
            return True
        except Exception:
            self._threeDVisible = False
            self._surfacePointCount = 0
            self._surfaceCellCount = 0
            self._setDetails("显示三维失败\n" + traceback.format_exc())
            if not silent:
                self._setResultMessage(
                    self.qcResultLabel,
                    "三维效果没有成功生成，请返回修改标注后重试。",
                    "warning",
                )
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
        if self._foregroundVoxelCount() <= 0:
            self._qcStatus = "需要确认"
            self._setResultMessage(
                self.qcResultLabel,
                "还没有标出髁突，暂时无法完成检查。",
                "warning",
            )
            self._setStatusMessage("还没有标出髁突，暂时无法完成检查。", "warning")
            self._syncUi()
            return False
        if not self._threeDVisible and not self._show3D(silent=True):
            self._qcStatus = "需要确认"
            self._syncUi()
            return False
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
            self._currentAnnotationStatus = "ANNOTATED"
            self._saved = True
            self._dirty = False
            self._annotationHasData = True
            self._qcStatus = "通过"
            completed = sum(
                self._caseStatusForIndex(index) in {"已标注", "已确认"}
                for index in range(len(self._caseFiles))
            )
            self.saveSuccessLabel.setVisible(True)
            self.saveSuccessLabel.setText(f"✓ {self._currentCaseId} 保存成功")
            self._setResultMessage(
                self.saveResultLabel,
                f"✓ {self._currentCaseId} 保存成功",
                "success",
            )
            if len(self._caseFiles) <= 1:
                nextGuidance = (
                    "这一例已经保存，但还没有确认。\n"
                    "请点击“确认本例标注”，确认后才能用于训练。\n"
                    "如果还有其它病例，可以继续导入和标注。\n"
                    "建议先完成至少 5 例进行第一次检查，之后再继续标更多病例。"
                )
            elif completed == len(self._caseFiles):
                nextGuidance = "所有病例都已保存；请逐例点击“确认本例标注”后再准备训练数据。"
            else:
                nextGuidance = (
                    f"已完成 {completed} / {len(self._caseFiles)} 例。\n"
                    "可以继续下一个病例。"
                )
            self._setResultMessage(
                self.saveNextGuidanceLabel, nextGuidance, "neutral"
            )
            self.nextCaseAfterSaveButton.setText(
                "继续下一个病例"
                if self._currentCaseIndex < len(self._caseFiles) - 1
                else "全部病例已经完成"
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

    def _confirmAnnotation(self):
        if not self._saved or self._dirty or not self._annotationHasData:
            self._setStatusMessage("请先保存并完成技术检查，再确认本例标注。", "warning")
            return False
        if self._currentAnnotationStatus == "VERIFIED":
            self._setStatusMessage("本例已经确认，可以用于训练。", "info")
            return True
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Question)
        box.setWindowTitle("确认本例标注")
        box.setText("确认后，这一例可以用于训练。")
        box.setInformativeText("请确认你已经完成医学复核，并且三维轮廓没有明显错误。")
        confirmButton = box.addButton("确认并用于训练", qt.QMessageBox.AcceptRole)
        cancelButton = box.addButton("取消", qt.QMessageBox.RejectRole)
        box.setDefaultButton(cancelButton)
        self._execDialog(box)
        if box.clickedButton() != confirmButton:
            return False
        try:
            self._upsertManifest(
                case_id=self._currentCaseId,
                volume=self.volumeNode,
                labelPath=self._outputPath(),
                status="VERIFIED",
                warnings=[],
            )
            self._currentAnnotationStatus = "VERIFIED"
            self._setResultMessage(
                self.saveResultLabel,
                f"✓ {self._currentCaseId} 已确认，可以用于训练",
                "success",
            )
            self._setResultMessage(
                self.saveNextGuidanceLabel,
                "本例已经进入可训练病例。可以继续确认其它病例，或返回首页查看下一步。",
                "success",
            )
            self.confirmAnnotationHintLabel.setText(
                "✓ 本例已确认，可以用于训练。重新编辑后需要再次确认。"
            )
            self._setStatusMessage("本例标注已确认，可以用于训练。", "success")
            self._syncUi()
            return True
        except Exception:
            self._setDetails("确认标注失败\n" + traceback.format_exc())
            self._setStatusMessage("确认状态没有保存，请重试。", "warning")
            return False

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
            "geometry_valid": "true" if status in {"ANNOTATED", "VERIFIED"} else existing.get("geometry_valid", ""),
            "label_valid": "true" if status in {"ANNOTATED", "VERIFIED"} else existing.get("label_valid", ""),
            "notes": "由下颌髁突三维分割实验平台保存。"
            + (" 已由用户确认。" if status == "VERIFIED" else "")
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
        self._threeDVisible = False
        self._surfacePointCount = 0
        self._surfaceCellCount = 0

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
        if self._process is not None:
            try:
                self._process.kill()
            except Exception:
                pass
            self._process = None
        if self._taskTimer is not None:
            try:
                self._taskTimer.stop()
            except Exception:
                pass
        self._clearResultComparison()
        self._removeNodeSafely(getattr(self, "_predictionImageNode", None))
        self._removeNodeSafely(self._predictionLabelNode)
        self._removeNodeSafely(self._predictionSegmentationNode)
        self._predictionLabelNode = None
        self._predictionSegmentationNode = None
        self._predictionImageNode = None
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
            return {
                "VERIFIED": "已确认",
                "UNVERIFIED": "未验证",
                "ANNOTATED": "已标注",
            }.get(
                self._currentAnnotationStatus,
                "未验证" if self._annotationHasData else "未标注",
            )
        if self._annotationHasData:
            return "标注中"
        if self._currentAnnotationStatus == "ANNOTATING":
            return "标注中"
        return "未标注"

    def _currentStep(self):
        if not self.volumeNode:
            return 0
        if self._homeVisible:
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
        if not self.volumeNode and not self._caseFiles:
            self.caseIdValue.setText("未选择")
            self.caseProgressValue.setText("—")
            self.caseStatusValue.setText("等待导入")
            self.caseSaveValue.setText("—")
        else:
            self.caseIdValue.setText(self._currentCaseId)
        if self._caseFiles and self._currentCaseIndex >= 0:
            progress = f"{self._currentCaseIndex + 1} / {len(self._caseFiles)}"
        elif self._caseFiles:
            progress = f"— / {len(self._caseFiles)}"
        else:
            progress = "1 / 1"
        if self.volumeNode or self._caseFiles:
            self.caseProgressValue.setText(progress)
            self.caseStatusValue.setText(status)
            self.caseSaveValue.setText(
                "已保存" if self._saved and not self._dirty else "未保存"
            )
        self.caseNavigationLabel.setText("病例 " + progress)

        if hasattr(self, "homeProgressLabel"):
            inventory, counts = self._inventoryAndCounts()
            self._refreshDerivedProjectState(inventory, counts)
            completed = int(counts["annotated"])
            verified = int(counts.get("verified", 0))
            total = int(counts["total"])
            self.homeProgressLabel.setText(
                f"病例进度：{total} 个病例　　人工标注完成：{completed}　　已确认：{verified}"
                if total
                else "还没有导入病例。"
            )
            self.homeStartButton.setEnabled(bool(self.volumeNode or self._caseFiles))
            self.homeContinueButton.setEnabled(True)
            self.homeProgressButton.setEnabled(bool(self._caseFiles))
            self.homeAnnotatedButton.setEnabled(True)
            self.homeTrainingButton.setEnabled(True)
            self.homeResultsButton.setEnabled(True)
            self.homePredictButton.setEnabled(True)
            self.homeStatLabels["total"].setText(str(total))
            self.homeStatLabels["manual"].setText(str(completed))
            self.homeStatLabels["verified"].setText(str(verified))
            self.homeStatLabels["trainable"].setText(str(counts["trainable"]))
            foldStates = detect_fold_states(
                fold_results_directory(
                    results_root=self.projectRoot / "workspace" / "nnUNet_results"
                )
            )
            modelReady = all(state.completed for state in foldStates)
            hasIncomplete = any(state.resumable for state in foldStates)
            self.homeStatLabels["training"].setText(
                "进行中" if self._processKind in {"training", "oof", "evaluation"}
                else "已完成" if modelReady
                else "未完成" if hasIncomplete
                else "未开始"
            )
            self.homeStatLabels["results"].setText(
                "可查看" if has_evaluation_results(self.projectRoot / "workspace" / "reports") else "暂无"
            )
            self.homeStatLabels["model"].setText(
                "已准备" if modelReady else "暂无"
            )
            action = self._homeNextAction(counts)
            self.homeNextStepLabel.setText(action.message)
            self.homeNextActionButton.setText(action.button)
            self.homeWorkflowLabel.setText(
                "病例：{manual} / {total} 已完成人工标注，{verified} 例已确认\n"
                "训练数据：{dataset}\n"
                "模型训练：{training}\n"
                "实验结果：{results}\n"
                "自动分割：{prediction}".format(
                    manual=completed,
                    total=total,
                    verified=verified,
                    dataset="已准备" if self._datasetPrepared else "未准备",
                    training=(
                        "进行中"
                        if self._processKind in {"training", "oof", "evaluation"}
                        else "已完成"
                        if modelReady
                        else "未完成"
                        if hasIncomplete
                        else "未开始"
                    ),
                    results="可查看" if has_evaluation_results(self.projectRoot / "workspace" / "reports") else "暂无",
                    prediction="可以使用" if modelReady else "等待模型",
                )
            )

        statusState = "complete" if status in {"已确认", "已标注"} else "working" if status == "标注中" else "warning" if status == "未验证" else ""
        statusPrefix = "✓ " if status in {"已确认", "已标注"} else "● "
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
        self.redisplay3DButton.setEnabled(hasSegmentation)
        self.checkButton.setEnabled(hasSegmentation)
        self.view2DButton.setEnabled(hasSegmentation)
        self.view3DButton.setEnabled(hasSegmentation)
        self.confirmCheckButton.setEnabled(self._qcStatus == "通过")
        self.saveButton.setEnabled(
            hasSegmentation and self._qcStatus == "通过"
        )
        self.confirmAnnotationButton.setEnabled(
            bool(self._saved and not self._dirty and self._annotationHasData)
            and self._currentAnnotationStatus != "VERIFIED"
        )
        self.confirmAnnotationButton.setText(
            "已确认本例标注"
            if self._currentAnnotationStatus == "VERIFIED"
            else "确认本例标注"
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
        self._refreshCaseListWidget()
        self._updatePageChrome()
        if self._mainPage == "prediction":
            self._refreshPredictionPage()

    def _homeNextStep(self, counts):
        return self._homeNextAction(counts).message

    def _homeNextAction(self, counts):
        states = detect_fold_states(
            fold_results_directory(results_root=self.projectRoot / "workspace" / "nnUNet_results")
        )
        return home_next_action(
            total_cases=int(counts.get("total", 0)),
            annotated_cases=int(counts.get("annotated", 0)),
            verified_cases=int(counts.get("verified", 0)),
            verified_group_count=int(counts.get("verified_group_count", 0)),
            validation_passed=self._validationPassed,
            dataset_prepared=self._datasetPrepared,
            training_active=self._processKind in {"training", "oof", "evaluation"},
            model_ready=all(state.completed for state in states),
            results_ready=has_evaluation_results(self.projectRoot / "workspace" / "reports"),
        )

    def _updateSaveSummary(self):
        if not hasattr(self, "saveSummaryLabel"):
            return
        self.saveSummaryLabel.setText(
            f"病例：{self._currentCaseId}\n"
                f"✓ 病例已导入\n"
                f"✓ 已完成标注\n"
                "✓ 技术检查完成\n"
                + ("✓ 本例已确认，可用于训练" if self._currentAnnotationStatus == "VERIFIED" else "○ 尚未确认，暂不能用于训练")
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
