import glob
import inspect
import logging
import os

import numpy as np
import qt
import ctk
import vtk

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleTest,
)


# -----------------------------------------------------------------------------
# Organ definitions
# -----------------------------------------------------------------------------
# Canonical labels follow TotalSegmentator's own naming convention
# (snake_case, e.g. "kidney_right"). These are only used as fallbacks and as
# fuzzy-matching keys: at runtime we prefer to read the *actual* class map of
# whichever TotalSegmentator task is installed, so the app stays correct even
# if label names change across TotalSegmentator versions.
#
# Both tasks below are the free/open TotalSegmentator models (Apache-2.0,
# no license key required): "total" for CT and "total_mr" for MRI.
NON_ORGAN_KEYWORDS = (
    "vertebrae", "rib", "hip", "femur", "humerus", "tibia", "fibula", "sacrum",
    "muscle", "gluteus", "iliopsoas", "autochthon", "fat", "skull", "artery",
    "vein", "aorta", "vena", "disc", "clavicle", "clavicula", "scapula", "sternum",
    "costal", "cava", "carpal", "tarsal", "patella", "vertebra", "spinal_cord",
    "skeletal", "atrial", "atrium", "ventricle", "pulmonary", "subclavian",
    "brachiocephalic", "common_carotid", "portal", "splenic", "cyst",
)

CT_ORGANS_FALLBACK = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach",
    "pancreas", "adrenal_gland_right", "adrenal_gland_left",
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", "trachea",
    "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder",
    "prostate", "heart", "brain",
]

MR_ORGANS_FALLBACK = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach",
    "pancreas", "adrenal_gland_right", "adrenal_gland_left", "lung_left",
    "lung_right", "esophagus", "small_bowel", "duodenum", "colon",
    "urinary_bladder", "prostate", "heart", "brain",
]

MODALITY_TASKS = {
    "CT": "total",
    "MRI": "total_mr",
}


def prettify_label(canonicalLabel):
    """'adrenal_gland_right' -> 'Right adrenal gland'"""
    parts = canonicalLabel.split("_")
    side = None
    if "left" in parts:
        parts.remove("left")
        side = "Left"
    elif "right" in parts:
        parts.remove("right")
        side = "Right"
    name = " ".join(parts)
    if side:
        return f"{side} {name}"
    return name[:1].upper() + name[1:]


def is_organ_label(canonicalLabel):
    lower = canonicalLabel.lower()
    return not any(keyword in lower for keyword in NON_ORGAN_KEYWORDS)


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------
class OrganAwareWindowing(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Organ-Aware Windowing"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Hackathon 2026"]
        self.parent.helpText = (
            "Select a folder with a CT or MRI volume (.nii/.nii.gz), pick an "
            "organ and modality, segment the organ with TotalSegmentator "
            "(free models only), optionally darken everything outside the organ "
            "in a new derived volume, then optimize the display window/level "
            "within that organ."
        )
        self.parent.acknowledgementText = ""


# -----------------------------------------------------------------------------
# Widget
# -----------------------------------------------------------------------------
class OrganAwareWindowingWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self.baseVolumeNode = None  # the originally loaded volume, unmodified
        self.currentVolumeNode = None  # whatever volume is currently displayed/optimized (base or suppressed)
        self._derivedVolumeNode = None  # the background-suppressed volume from the last Optimize Contrast click, if any
        self.currentSegmentationNode = None
        self.currentSegmentId = None
        self.currentOrganLabel = None
        self._trackedNodes = []  # every node created by this widget, for cleanup on the next run

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = OrganAwareWindowingLogic()

        # ---------------- Input section ----------------
        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Input"
        self.layout.addWidget(inputCollapsible)
        inputForm = qt.QFormLayout(inputCollapsible)

        self.directoryButton = ctk.ctkDirectoryButton()
        self.directoryButton.directory = qt.QDir.homePath()
        inputForm.addRow("Image folder:", self.directoryButton)

        self.fileComboBox = qt.QComboBox()
        self.fileComboBox.enabled = False
        inputForm.addRow("Volume file:", self.fileComboBox)

        self.modalityComboBox = qt.QComboBox()
        self.modalityComboBox.addItems(list(MODALITY_TASKS.keys()))
        inputForm.addRow("Modality:", self.modalityComboBox)

        self.organComboBox = qt.QComboBox()
        inputForm.addRow("Organ:", self.organComboBox)

        # ---------------- Segmentation section ----------------
        segCollapsible = ctk.ctkCollapsibleButton()
        segCollapsible.text = "Segmentation"
        self.layout.addWidget(segCollapsible)
        segLayout = qt.QVBoxLayout(segCollapsible)

        self.segmentButton = qt.QPushButton("Segment Organ")
        self.segmentButton.enabled = False
        segLayout.addWidget(self.segmentButton)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        segLayout.addWidget(self.statusLabel)

        # ---------------- Contrast section ----------------
        contrastCollapsible = ctk.ctkCollapsibleButton()
        contrastCollapsible.text = "Contrast"
        self.layout.addWidget(contrastCollapsible)
        contrastLayout = qt.QVBoxLayout(contrastCollapsible)

        contrastForm = qt.QFormLayout()
        self.suppressionSlider = ctk.ctkSliderWidget()
        self.suppressionSlider.minimum = 0
        self.suppressionSlider.maximum = 5
        self.suppressionSlider.value = 2.5
        self.suppressionSlider.singleStep = 0.1
        self.suppressionSlider.decimals = 1
        self.suppressionSlider.suffix = " %"
        self.suppressionSlider.toolTip = (
            "How strongly to darken everything outside the segmented organ, toward the "
            "volume's own darkest voxel value, before computing the optimized window/level. "
            "0% leaves intensities unchanged. Applied to a new derived volume when "
            "'Optimize Contrast' is clicked — the original loaded volume is left untouched."
        )
        contrastForm.addRow("Background suppression:", self.suppressionSlider)
        contrastLayout.addLayout(contrastForm)

        self.optimizeButton = qt.QPushButton("Optimize Contrast")
        self.optimizeButton.enabled = False
        contrastLayout.addWidget(self.optimizeButton)

        self.contrastLabel = qt.QLabel("")
        self.contrastLabel.wordWrap = True
        contrastLayout.addWidget(self.contrastLabel)

        self.saveSceneButton = qt.QPushButton("Save Scene (.mrb)...")
        self.saveSceneButton.enabled = False
        self.saveSceneButton.toolTip = (
            "Save the volume, segmentation, and the optimized window/level as a single "
            "Slicer Data Bundle. Reopening this .mrb restores the same display settings — "
            "reopening the original .nii/.nii.gz directly will not, since NIfTI has no "
            "field to store window/level."
        )
        contrastLayout.addWidget(self.saveSceneButton)

        self.layout.addStretch(1)

        # ---------------- Signals ----------------
        self.directoryButton.directoryChanged.connect(self.onDirectoryChanged)
        self.fileComboBox.currentIndexChanged.connect(self.onSelectionChanged)
        self.modalityComboBox.currentIndexChanged.connect(self.onModalityChanged)
        self.organComboBox.currentIndexChanged.connect(self.onSelectionChanged)
        self.segmentButton.clicked.connect(self.onSegmentClicked)
        self.optimizeButton.clicked.connect(self.onOptimizeClicked)
        self.saveSceneButton.clicked.connect(self.onSaveSceneClicked)

        self.onModalityChanged()
        self.onDirectoryChanged(self.directoryButton.directory)

    # -------------------------------------------------------------------
    def onDirectoryChanged(self, directory):
        self.fileComboBox.clear()
        files = self.logic.findNiftiFiles(directory)
        for f in files:
            self.fileComboBox.addItem(os.path.basename(f), f)
        self.fileComboBox.enabled = len(files) > 0
        self.onSelectionChanged()

    def onModalityChanged(self):
        modality = self.modalityComboBox.currentText
        task = MODALITY_TASKS[modality]
        organLabels = self.logic.getOrganLabels(task)
        self.organComboBox.clear()
        for label in organLabels:
            self.organComboBox.addItem(prettify_label(label), label)
        self.onSelectionChanged()

    def onSelectionChanged(self):
        ready = self.fileComboBox.enabled and self.fileComboBox.count > 0 and self.organComboBox.count > 0
        self.segmentButton.enabled = ready
        self.optimizeButton.enabled = False
        self.saveSceneButton.enabled = False
        self.contrastLabel.text = ""

    # -------------------------------------------------------------------
    def onSegmentClicked(self):
        filePath = self.fileComboBox.itemData(self.fileComboBox.currentIndex)
        modality = self.modalityComboBox.currentText
        task = MODALITY_TASKS[modality]
        organLabel = self.organComboBox.itemData(self.organComboBox.currentIndex)
        organName = self.organComboBox.currentText

        if not filePath:
            slicer.util.errorDisplay("Please select a folder containing a .nii or .nii.gz file.")
            return

        self.segmentButton.enabled = False
        self.optimizeButton.enabled = False
        self.saveSceneButton.enabled = False
        self.contrastLabel.text = ""
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._clearPreviousResults()

            self.statusLabel.text = f"Loading {os.path.basename(filePath)}..."
            slicer.app.processEvents()
            self.baseVolumeNode = self._track(self.logic.loadVolume(filePath))
            self.currentVolumeNode = self.baseVolumeNode
            slicer.util.setSliceViewerLayers(background=self.currentVolumeNode, fit=True)

            self.statusLabel.text = (
                f"Running TotalSegmentator ({task}) for '{organName}'. "
                "First run may download the free model weights and can take "
                "several minutes..."
            )
            slicer.app.processEvents()

            segmentationNode = self._track(self.logic.runSegmentation(self.currentVolumeNode, task))

            segmentId = self.logic.isolateOrgan(segmentationNode, organLabel)
            if segmentId is None:
                slicer.util.errorDisplay(
                    f"'{organName}' was not found among the structures produced by the "
                    f"'{task}' TotalSegmentator model. Try a different organ or modality."
                )
                self.statusLabel.text = "Segmentation finished, but organ was not found."
                return

            self.currentSegmentationNode = segmentationNode
            self.currentSegmentId = segmentId
            self.currentOrganLabel = organLabel

            self.statusLabel.text = f"Segmented '{organName}'. Ready to optimize contrast."
            self.optimizeButton.enabled = True
        except Exception as e:
            logging.exception("Segmentation failed")
            slicer.util.errorDisplay(f"Segmentation failed: {e}")
            self.statusLabel.text = "Segmentation failed."
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.segmentButton.enabled = True

    def onOptimizeClicked(self):
        if not (self.baseVolumeNode and self.currentSegmentationNode and self.currentSegmentId):
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            # Drop any derived (suppressed) volume from a previous click so repeated
            # clicks don't stack suppression on top of suppression.
            if self._derivedVolumeNode is not None and slicer.mrmlScene.IsNodePresent(self._derivedVolumeNode):
                slicer.mrmlScene.RemoveNode(self._derivedVolumeNode)
                self._trackedNodes = [n for n in self._trackedNodes if n is not self._derivedVolumeNode]
            self._derivedVolumeNode = None

            fraction = self.suppressionSlider.value / 100.0
            if fraction > 0:
                newVolumeNode = self._track(self.logic.createBackgroundSuppressedVolume(
                    self.baseVolumeNode, self.currentSegmentationNode, self.currentSegmentId, fraction
                ))
                self._derivedVolumeNode = newVolumeNode
                self.currentVolumeNode = newVolumeNode
                self.currentSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(newVolumeNode)
            else:
                self.currentVolumeNode = self.baseVolumeNode
            slicer.util.setSliceViewerLayers(background=self.currentVolumeNode, fit=False)

            window, level = self.logic.optimizeContrast(
                self.currentVolumeNode, self.currentSegmentationNode, self.currentSegmentId
            )
            segmentationDisplayNode = self.currentSegmentationNode.GetDisplayNode()
            if segmentationDisplayNode is not None:
                segmentationDisplayNode.SetVisibility(False)

            centroidRAS = self.logic.getSegmentCentroidRAS(
                self.currentVolumeNode, self.currentSegmentationNode, self.currentSegmentId
            )
            if centroidRAS is not None:
                slicer.modules.markups.logic().JumpSlicesToLocation(
                    centroidRAS[0], centroidRAS[1], centroidRAS[2], True
                )

            self.contrastLabel.text = (
                f"Background suppression: {self.suppressionSlider.value:.1f}%. "
                f"Optimized window/level: {window:.1f} / {level:.1f}"
            )
            self.saveSceneButton.enabled = True
        except Exception as e:
            logging.exception("Contrast optimization failed")
            slicer.util.errorDisplay(f"Contrast optimization failed: {e}")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onSaveSceneClicked(self):
        if not self.currentVolumeNode:
            return

        sourcePath = self.fileComboBox.itemData(self.fileComboBox.currentIndex) or ""
        sourceDir = os.path.dirname(sourcePath) if sourcePath else qt.QDir.homePath()
        baseName = os.path.splitext(os.path.splitext(os.path.basename(sourcePath))[0])[0] if sourcePath else "scene"
        organSuffix = self.currentOrganLabel or "organ"
        defaultPath = os.path.join(sourceDir, f"{baseName}_{organSuffix}_optimized.mrb")

        savePath = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(), "Save Scene", defaultPath, "Slicer Data Bundle (*.mrb)"
        )
        if not savePath:
            return
        if not savePath.lower().endswith(".mrb"):
            savePath += ".mrb"

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            success = slicer.util.saveScene(savePath)
            if success:
                self.statusLabel.text = f"Scene saved to {savePath}. Reopen this file to restore the optimized window/level."
            else:
                slicer.util.errorDisplay(f"Failed to save scene to {savePath}. See the application log for details.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def _track(self, node):
        self._trackedNodes.append(node)
        return node

    def _clearPreviousResults(self):
        for node in self._trackedNodes:
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._trackedNodes = []
        self.baseVolumeNode = None
        self.currentVolumeNode = None
        self._derivedVolumeNode = None
        self.currentSegmentationNode = None
        self.currentSegmentId = None
        self.currentOrganLabel = None


# -----------------------------------------------------------------------------
# Logic
# -----------------------------------------------------------------------------
class OrganAwareWindowingLogic(ScriptedLoadableModuleLogic):

    def findNiftiFiles(self, folderPath):
        if not folderPath or not os.path.isdir(folderPath):
            return []
        files = []
        for pattern in ("*.nii.gz", "*.nii"):
            files.extend(glob.glob(os.path.join(folderPath, pattern)))
        return sorted(set(files))

    def getOrganLabels(self, task):
        """Prefer the real, installed TotalSegmentator class map; fall back
        to a curated static list if the package isn't installed yet."""
        try:
            from totalsegmentator.map_to_binary import class_map
            labels = list(class_map.get(task, {}).values())
            if labels:
                return sorted(l for l in labels if is_organ_label(l))
        except Exception:
            pass
        fallback = CT_ORGANS_FALLBACK if task == "total" else MR_ORGANS_FALLBACK
        return sorted(fallback)

    def loadVolume(self, filePath):
        return slicer.util.loadVolume(filePath)

    def _getTotalSegmentatorLogic(self):
        try:
            import TotalSegmentator
        except ImportError:
            raise RuntimeError(
                "The 'TotalSegmentator' extension is not installed. Install it from "
                "View > Extensions Manager (search for 'TotalSegmentator'), then restart Slicer."
            )
        return TotalSegmentator.TotalSegmentatorLogic()

    def runSegmentation(self, inputVolume, task):
        tsLogic = self._getTotalSegmentatorLogic()

        if hasattr(tsLogic, "setupPythonRequirements"):
            tsLogic.setupPythonRequirements()

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(slicer.mrmlScene.GenerateUniqueName(f"{task}_segmentation"))
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        sig = inspect.signature(tsLogic.process)
        kwargs = {}
        if "quality" in sig.parameters:
            kwargs["quality"] = "fast"
        elif "fast" in sig.parameters:
            kwargs["fast"] = True
        if "task" in sig.parameters:
            kwargs["task"] = task

        tsLogic.process(inputVolume, segmentationNode, **kwargs)

        try:
            segmentationNode.CreateClosedSurfaceRepresentation()
        except Exception:
            pass

        return segmentationNode

    def isolateOrgan(self, segmentationNode, canonicalLabel):
        """Find the segment matching canonicalLabel, hide all others, and
        return its segment ID (or None if not found)."""
        segmentation = segmentationNode.GetSegmentation()
        segmentIds = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]

        matchedId = self._findSegmentId(segmentation, segmentIds, canonicalLabel)
        if matchedId is None:
            return None

        displayNode = segmentationNode.GetDisplayNode()
        if displayNode is not None:
            for sid in segmentIds:
                displayNode.SetSegmentVisibility(sid, sid == matchedId)
        return matchedId

    @staticmethod
    def _findSegmentId(segmentation, segmentIds, canonicalLabel):
        if canonicalLabel in segmentIds:
            return canonicalLabel
        lowerMap = {sid.lower(): sid for sid in segmentIds}
        if canonicalLabel.lower() in lowerMap:
            return lowerMap[canonicalLabel.lower()]

        tokens = [t for t in canonicalLabel.lower().split("_") if t]

        for sid in segmentIds:
            segment = segmentation.GetSegment(sid)
            normalized = segment.GetName().lower().replace("-", " ").replace("_", " ")
            if all(tok in normalized for tok in tokens):
                return sid

        for sid in segmentIds:
            normalized = sid.lower().replace("-", " ").replace("_", " ")
            if all(tok in normalized for tok in tokens):
                return sid

        return None

    def optimizeContrast(self, volumeNode, segmentationNode, segmentId):
        labelmapArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
        volumeArray = slicer.util.arrayFromVolume(volumeNode)

        if labelmapArray.shape != volumeArray.shape:
            raise RuntimeError("Segmentation and volume geometries do not match.")

        values = volumeArray[labelmapArray > 0]
        if values.size == 0:
            raise RuntimeError("The selected organ segment is empty.")

        low, high = np.percentile(values, [1, 99])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high <= low:
            high = low + 1.0

        window = float(high - low)
        level = float((high + low) / 2.0)

        displayNode = volumeNode.GetDisplayNode()
        displayNode.SetAutoWindowLevel(False)
        displayNode.SetWindow(window)
        displayNode.SetLevel(level)

        slicer.util.forceRenderAllViews()
        return window, level

    def getSegmentCentroidRAS(self, volumeNode, segmentationNode, segmentId):
        """Returns the (R, A, S) centroid of the segment's voxels, or None if empty."""
        labelmapArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
        mask = labelmapArray > 0
        if not mask.any():
            return None

        # arrayFromSegmentBinaryLabelmap/arrayFromVolume index as [k, j, i] (z, y, x);
        # IJKToRASMatrix expects (i, j, k), so reverse the averaged index.
        kji = np.argwhere(mask).mean(axis=0)
        ijk = kji[::-1]

        ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRAS)
        ras = ijkToRAS.MultiplyPoint([float(ijk[0]), float(ijk[1]), float(ijk[2]), 1.0])
        return ras[0], ras[1], ras[2]

    def createBackgroundSuppressedVolume(self, volumeNode, segmentationNode, segmentId, suppressionFraction=0.8):
        """Returns a new volume node with the same geometry as volumeNode,
        where voxels *outside* the organ segment have their intensity pulled
        toward the source volume's own minimum value (i.e. lowered / darkened),
        by suppressionFraction (0 = unchanged, 1 = fully flattened to the
        minimum). Voxels inside the organ are left exactly as they were.
        The source volumeNode itself is not modified.
        """
        labelmapArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
        volumeArray = slicer.util.arrayFromVolume(volumeNode)

        if labelmapArray.shape != volumeArray.shape:
            raise RuntimeError("Segmentation and volume geometries do not match.")

        organMask = labelmapArray > 0
        if not organMask.any():
            raise RuntimeError("The selected organ segment is empty.")

        floorValue = float(volumeArray.min())
        suppressed = volumeArray.astype(np.float64, copy=True)
        background = ~organMask
        suppressed[background] = (
            suppressed[background] * (1.0 - suppressionFraction) + floorValue * suppressionFraction
        )

        ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRAS)
        newName = slicer.mrmlScene.GenerateUniqueName(f"{volumeNode.GetName()}_organ_isolated")
        newVolumeNode = slicer.util.addVolumeFromArray(suppressed, ijkToRAS=ijkToRAS, name=newName)
        return newVolumeNode


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
class OrganAwareWindowingTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_OptimizeContrastMath()

    def test_OptimizeContrastMath(self):
        """Validates the contrast-optimization math without requiring
        TotalSegmentator or network access."""
        self.delayDisplay("Starting contrast optimization test")

        size = (20, 20, 20)
        volumeArray = np.random.uniform(0, 100, size).astype(np.float32)
        volumeArray[5:10, 5:10, 5:10] = np.random.uniform(400, 500, (5, 5, 5)).astype(np.float32)

        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        slicer.util.updateVolumeFromArray(volumeNode, volumeArray)
        volumeNode.CreateDefaultDisplayNodes()

        labelArray = np.zeros(size, dtype=np.uint8)
        labelArray[5:10, 5:10, 5:10] = 1
        labelVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(labelVolumeNode, labelArray)
        labelVolumeNode.CopyOrientation(volumeNode)

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelVolumeNode, segmentationNode)
        segmentId = segmentationNode.GetSegmentation().GetNthSegmentID(0)

        logic = OrganAwareWindowingLogic()
        window, level = logic.optimizeContrast(volumeNode, segmentationNode, segmentId)

        self.assertTrue(300 < level < 600)
        self.assertTrue(window > 0)

        self.delayDisplay("Test passed")
