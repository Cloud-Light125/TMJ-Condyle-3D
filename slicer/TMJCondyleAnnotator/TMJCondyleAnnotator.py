"""3D Slicer helper module for one-class mandibular-condyle annotation.

This module intentionally delegates painting, erasing, drawing, scissors,
islands, fill-between-slices, and 3D rendering to Slicer's native
Segment Editor and segmentation display. It only adds a small, task-specific
workflow and strict export QC.
"""

from __future__ import annotations

import csv
import re
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


class TMJCondyleAnnotator(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "TMJ Condyle Annotator"
        self.parent.categories = ["Segmentation"]
        self.parent.contributors = ["TMJ-Condyle-3D"]
        self.parent.helpText = (
            "A small workflow wrapper around the native 3D Slicer Segment Editor "
            "for one class: mandibular condyle."
        )
        self.parent.acknowledgmentText = (
            "This module is part of the TMJ-Condyle-3D teaching/research workflow."
        )


class TMJCondyleAnnotatorWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.volumeNode = None
        self.segmentationNode = None
        self.segmentId = None
        self.projectRoot = Path(__file__).resolve().parents[2]

        intro = qt.QLabel(
            "<b>TMJ Condyle Annotator</b><br>"
            "只创建一个结构：Mandibular Condyle（label value 1）。"
            "绘制工具由 Slicer 原生 Segment Editor 提供。"
        )
        intro.wordWrap = True
        self.layout.addWidget(intro)

        volume_form = qt.QFormLayout()
        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = False
        self.volumeSelector.showHidden = False
        self.volumeSelector.showChildNodeTypes = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.currentNodeChanged.connect(self._onVolumeChanged)
        volume_form.addRow("MRI volume", self.volumeSelector)
        self.layout.addLayout(volume_form)

        load_button = qt.QPushButton("加载 MRI 文件")
        load_button.toolTip = "加载 .nii.gz/.nii/.nrrd；DICOM 请先用 Slicer DICOM 模块导入。"
        load_button.clicked.connect(self._loadVolume)
        self.layout.addWidget(load_button)

        case_form = qt.QFormLayout()
        self.caseIdEdit = qt.QLineEdit("case_001")
        self.caseIdEdit.toolTip = "只能填写匿名 case id，例如 case_001；不要填写姓名或 PatientID。"
        case_form.addRow("匿名 case id", self.caseIdEdit)
        self.outputEdit = qt.QLineEdit(
            str(self.projectRoot / "workspace" / "labels" / "case_001.nii.gz")
        )
        output_row = qt.QHBoxLayout()
        output_row.addWidget(self.outputEdit)
        output_browse = qt.QPushButton("选择")
        output_browse.clicked.connect(self._chooseOutput)
        output_row.addWidget(output_browse)
        case_form.addRow("Mask 输出路径", output_row)
        self.manifestEdit = qt.QLineEdit("")
        manifest_browse = qt.QPushButton("选择")
        manifest_browse.clicked.connect(self._chooseManifest)
        manifest_row = qt.QHBoxLayout()
        manifest_row.addWidget(self.manifestEdit)
        manifest_row.addWidget(manifest_browse)
        case_form.addRow("dataset_manifest.csv（可选）", manifest_row)
        self.layout.addLayout(case_form)

        self.createButton = qt.QPushButton("创建下颌髁突标注")
        self.createButton.toolTip = "创建唯一 segment，并把当前 MRI 设置为 reference geometry。"
        self.createButton.clicked.connect(self._createSegmentation)
        self.layout.addWidget(self.createButton)

        self.startButton = qt.QPushButton("开始标注（打开 Segment Editor）")
        self.startButton.clicked.connect(self._startAnnotation)
        self.layout.addWidget(self.startButton)

        action_row = qt.QHBoxLayout()
        self.show3DButton = qt.QPushButton("Show 3D")
        self.show3DButton.clicked.connect(self._show3D)
        action_row.addWidget(self.show3DButton)
        self.saveButton = qt.QPushButton("保存髁突 Mask")
        self.saveButton.clicked.connect(self._saveMask)
        action_row.addWidget(self.saveButton)
        self.layout.addLayout(action_row)

        self.statusLabel = qt.QLabel("状态：请加载一个 MRI volume。")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)
        self.qcText = qt.QPlainTextEdit()
        self.qcText.readOnly = True
        self.qcText.minimumHeight = 120
        self.layout.addWidget(self.qcText)
        self.layout.addStretch(1)

    def _setStatus(self, text: str):
        self.statusLabel.setText(f"状态：{text}")

    def _onVolumeChanged(self, node):
        self.volumeNode = node
        if node:
            self._setStatus(f"已选择 {node.GetName()}；请创建髁突标注。")

    def _loadVolume(self):
        file_name = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "加载 MRI",
            "",
            "Medical image (*.nii.gz *.nii *.nrrd);;All files (*)",
        )
        if isinstance(file_name, tuple):
            file_name = file_name[0]
        if not file_name:
            return
        node = slicer.util.loadVolume(file_name)
        if not node:
            slicer.util.errorDisplay(f"无法加载 MRI：{file_name}")
            return
        self.volumeSelector.setCurrentNode(node)
        self.volumeNode = node
        self._setStatus(f"已加载 {Path(file_name).name}；请创建髁突标注。")

    def _chooseOutput(self):
        file_name = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "保存髁突 Mask",
            self.outputEdit.text,
            "NIfTI label (*.nii.gz);;All files (*)",
        )
        if isinstance(file_name, tuple):
            file_name = file_name[0]
        if file_name:
            self.outputEdit.setText(file_name)

    def _chooseManifest(self):
        file_name = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "选择私有 dataset_manifest.csv",
            self.manifestEdit.text,
            "CSV (*.csv);;All files (*)",
        )
        if isinstance(file_name, tuple):
            file_name = file_name[0]
        if file_name:
            self.manifestEdit.setText(file_name)

    def _requireVolume(self):
        node = self.volumeSelector.currentNode() or self.volumeNode
        if not node:
            slicer.util.warningDisplay("请先选择或加载 MRI volume。")
            return None
        self.volumeNode = node
        return node

    def _createSegmentation(self):
        volume = self._requireVolume()
        if not volume:
            return
        if not self.segmentationNode:
            self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "TMJ_Condyle_Segmentation"
            )
            self.segmentationNode.CreateDefaultDisplayNodes()
        self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volume)
        segmentation = self.segmentationNode.GetSegmentation()
        self.segmentId = None
        for index in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(index)
            if segmentation.GetSegment(segment_id).GetName() == "Mandibular Condyle":
                self.segmentId = segment_id
                break
        if not self.segmentId:
            self.segmentId = segmentation.AddEmptySegment(
                "Mandibular Condyle",
                "MandibularCondyle",
                (0.95, 0.45, 0.10),
            )
        self.segmentationNode.SetDisplayVisibility(True)
        manifest_text = str(self.manifestEdit.text).strip()
        case_id = str(self.caseIdEdit.text).strip()
        if manifest_text and CASE_ID_PATTERN.fullmatch(case_id):
            self._upsertManifest(
                Path(manifest_text),
                case_id=case_id,
                volume=volume,
                label_path=None,
                status="ANNOTATING",
                warnings=[],
            )
        elif manifest_text:
            slicer.util.warningDisplay(
                "case_id 不是匿名格式，未更新 manifest；请使用 case_001 形式。"
            )
        self._setStatus(
            "已创建唯一 segment：Mandibular Condyle。点击“开始标注”进入 Segment Editor。"
        )

    def _startAnnotation(self):
        if not self.segmentationNode:
            self._createSegmentation()
        volume = self._requireVolume()
        if not volume or not self.segmentationNode:
            return
        slicer.util.selectModule("SegmentEditor")
        editor_widget = slicer.util.getModuleWidget("SegmentEditor")
        if editor_widget is None:
            slicer.util.errorDisplay("无法取得 Segment Editor。请确认 Slicer 的 Segmentations 模块可用。")
            return
        editor_widget.setSegmentationNode(self.segmentationNode)
        editor_widget.setMasterVolumeNode(volume)
        if self.segmentId:
            editor_widget.setCurrentSegmentID(self.segmentId)
        self._setStatus(
            "Segment Editor 已打开。请用 Paint/Erase/Draw/Scissors 等工具逐层检查髁突。"
        )

    def _show3D(self):
        if not self.segmentationNode:
            slicer.util.warningDisplay("请先创建髁突标注。")
            return
        self.segmentationNode.CreateClosedSurfaceRepresentation()
        display_node = self.segmentationNode.GetDisplayNode()
        display_node.SetVisibility(True)
        display_node.SetVisibility3D(True)
        display_node.SetVisibility2DFill(True)
        display_node.SetVisibility2DOutline(True)
        self._setStatus("已打开 3D 显示。请旋转表面检查是否有明显漏标或误标。")

    @staticmethod
    def _node_geometry(node):
        dimensions = tuple(int(value) for value in node.GetImageData().GetDimensions())
        spacing = tuple(float(value) for value in node.GetSpacing())
        origin = tuple(float(value) for value in node.GetOrigin())
        matrix = vtk.vtkMatrix3x3()
        try:
            node.GetIJKToRASDirectionMatrix(matrix)
            direction = tuple(float(matrix.GetElement(i, j)) for i in range(3) for j in range(3))
        except Exception:
            direction = ()
        return dimensions, spacing, origin, direction

    @classmethod
    def _geometry_errors(cls, image_node, label_node):
        image = cls._node_geometry(image_node)
        label = cls._node_geometry(label_node)
        errors = []
        if image[0] != label[0]:
            errors.append(f"shape {image[0]} != {label[0]}")
        for name, left, right in (
            ("spacing", image[1], label[1]),
            ("origin", image[2], label[2]),
            ("direction", image[3], label[3]),
        ):
            if len(left) != len(right) or any(abs(a - b) > 1e-4 for a, b in zip(left, right)):
                errors.append(f"{name} differs")
        return errors

    @staticmethod
    def _component_warning(mask):
        try:
            from scipy import ndimage
        except ImportError:
            return None
        structure = ndimage.generate_binary_structure(mask.ndim, 1)
        _, count = ndimage.label(mask, structure=structure)
        return int(count)

    def _saveMask(self):
        volume = self._requireVolume()
        if not volume or not self.segmentationNode:
            slicer.util.warningDisplay("请先加载 MRI 并创建/完成髁突标注。")
            return
        case_id = str(self.caseIdEdit.text).strip()
        if not CASE_ID_PATTERN.fullmatch(case_id):
            slicer.util.errorDisplay(
                "case_id 不合规。请使用匿名格式，例如 case_001；不要使用姓名或 PatientID。"
            )
            return
        output_text = str(self.outputEdit.text).strip()
        if not output_text:
            slicer.util.errorDisplay("请先选择 mask 输出路径。")
            return
        output_path = Path(output_text)
        if not output_path.name.endswith(".nii.gz"):
            output_path = output_path.with_name(output_path.name + ".nii.gz")
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            answer = qt.QMessageBox.question(
                slicer.util.mainWindow(),
                "覆盖已生成文件？",
                f"输出已存在：{output_path}\n只允许覆盖生成的 mask，不会修改原始 MRI。继续？",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
            )
            if answer != qt.QMessageBox.Yes:
                return

        segmentation = self.segmentationNode.GetSegmentation()
        if segmentation.GetNumberOfSegments() != 1:
            slicer.util.errorDisplay("标注必须只包含一个 segment：Mandibular Condyle。")
            return
        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", f"{case_id}_condyle_label"
        )
        try:
            labelmap.SetReferenceImageGeometryParameterFromVolumeNode(volume)
            logic = slicer.modules.segmentations.logic()
            if hasattr(logic, "ExportAllSegmentsToLabelmapNode"):
                ok = logic.ExportAllSegmentsToLabelmapNode(
                    self.segmentationNode, labelmap, volume
                )
            else:
                ok = logic.ExportVisibleSegmentsToLabelmapNode(
                    self.segmentationNode, labelmap, volume
                )
            if ok is False:
                raise RuntimeError("Slicer segmentation export returned false")
            geometry_errors = self._geometry_errors(volume, labelmap)
            if geometry_errors:
                raise RuntimeError("geometry mismatch: " + "; ".join(geometry_errors))
            array = np.asarray(slicer.util.arrayFromVolume(labelmap))
            if not np.isfinite(array).all():
                raise RuntimeError("mask contains NaN or Inf")
            if not np.equal(array, np.floor(array)).all():
                raise RuntimeError("mask contains non-integer labels")
            values = sorted(int(value) for value in np.unique(array))
            if any(value not in (0, 1) for value in values):
                raise RuntimeError(f"mask labels must be 0/1; found {values}")
            foreground = int(np.count_nonzero(array == 1))
            if foreground <= 0:
                raise RuntimeError("mask is empty; empty labels cannot be saved as completed")
            spacing = tuple(float(value) for value in volume.GetSpacing())
            physical_volume = foreground * float(np.prod(spacing))
            components = self._component_warning(array == 1)
            warnings = []
            if components is not None and components > 10:
                warnings.append(
                    f"发现 {components} 个 connected components，请回到三视图检查小孤立区域。"
                )
            if not slicer.util.saveNode(labelmap, str(output_path)):
                raise RuntimeError("Slicer could not save the label map")
            self.qcText.setPlainText(
                "\n".join(
                    [
                        "导出 QC：PASS",
                        "labels: {0, 1}",
                        f"foreground voxels: {foreground}",
                        f"physical volume: {physical_volume:.3f} mm³",
                        f"connected components: {components if components is not None else 'not measured'}",
                        *warnings,
                    ]
                )
            )
            manifest_text = str(self.manifestEdit.text).strip()
            if manifest_text:
                self._upsertManifest(
                    Path(manifest_text),
                    case_id=case_id,
                    volume=volume,
                    label_path=output_path,
                    status="ANNOTATED",
                    warnings=warnings,
                )
            self._setStatus(f"Mask 已保存：{output_path}；可以交给 validate_case.py 做第二次 QC。")
        except Exception as exc:
            slicer.util.errorDisplay(f"Mask 导出被拒绝：{exc}")
            self.qcText.setPlainText(f"导出 QC：FAIL\n{type(exc).__name__}: {exc}")
        finally:
            slicer.mrmlScene.RemoveNode(labelmap)

    @staticmethod
    def _image_path_for_volume(volume):
        storage = volume.GetStorageNode()
        if storage and storage.GetFileName():
            path = Path(storage.GetFileName()).resolve()
            project_root = Path(__file__).resolve().parents[2]
            try:
                return path.relative_to(project_root).as_posix()
            except ValueError:
                return ""
        return ""

    def _manifest_path(self, path):
        if not path:
            return ""
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.projectRoot).as_posix()
        except ValueError:
            return ""

    def _upsertManifest(self, manifest_path, *, case_id, volume, label_path, status, warnings):
        rows = []
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        existing = next((row for row in rows if row.get("case_id") == case_id), {})
        image_path = self._image_path_for_volume(volume) or existing.get("image_path", "")
        label_value = self._manifest_path(label_path) if label_path else existing.get("label_path", "")
        updated = {
            "case_id": case_id,
            "group_id": existing.get("group_id") or case_id,
            "side": existing.get("side", ""),
            "image_path": image_path,
            "label_path": label_value,
            "annotation_status": status,
            "geometry_valid": "true" if status == "ANNOTATED" else existing.get("geometry_valid", ""),
            "label_valid": "true" if status == "ANNOTATED" else existing.get("label_valid", ""),
            "notes": (
                "Exported by TMJ Condyle Annotator; " + " ".join(warnings)
                if status == "ANNOTATED"
                else existing.get("notes", "")
            ),
        }
        rows = [row for row in rows if row.get("case_id") != case_id]
        rows.append(updated)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in MANIFEST_FIELDS} for row in rows)
        if not image_path:
            slicer.util.warningDisplay(
                "当前 MRI 不在项目 workspace 内，manifest 未写入 image_path；请先放入 workspace/nifti 再训练。"
            )


class TMJCondyleAnnotatorTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_case_id_and_binary_rules()

    def test_case_id_and_binary_rules(self):
        self.assertTrue(CASE_ID_PATTERN.fullmatch("case_001"))
        self.assertFalse(CASE_ID_PATTERN.fullmatch("患者张三"))
        self.assertTrue(set(np.unique(np.array([0, 1], dtype=np.uint8))).issubset({0, 1}))
