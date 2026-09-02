# Development notes

## Scope

The repository owns data conversion, manifest/QC, nnU-Net v2 dataset preparation,
split generation, training orchestration, OOF evaluation, inference orchestration,
model export, and the small Slicer workflow wrapper. It does not own a new neural
network or a replacement image viewer.

## Third-party boundaries

Temporary reference clones live below references/ and are ignored. Do not copy source
files from those repositories into this project. If an implementation idea is used,
rewrite it against this project's task definition and cite the upstream project.

## Code conventions

- Keep the label vocabulary exactly 0/background and 1/mandibular_condyle.
- Treat SimpleITK geometry as x-y-z; treat NumPy array axes as z-y-x.
- Never sort DICOM filenames manually.
- Never silently resample, resize, pad, or relabel a mask in QC.
- Use grouped splits and make leakage checks fail loudly.
- Keep all CLI scripts runnable as python scripts/name.py from the project root.
- Do not log or serialize DICOM identifiers.

## Validation

Run only checks relevant to the change:

~~~powershell
python -m compileall tmj_condyle scripts
python -m pytest -q
python scripts/scan_git_safety.py
~~~

Slicer module tests need to run inside a compatible 3D Slicer instance and are not
covered by the ordinary Python venv.

## Release gate

Before a public commit:

1. inspect git status, diff, and tracked file list;
2. run scan_git_safety.py on staged files;
3. verify workspace and references remain ignored;
4. verify no DICOM/NIfTI/mask/checkpoint is staged;
5. record actual Python/PyTorch/nnU-Net/SlicerNNUnet versions;
6. mark unrun clinical-data steps as NOT RUN, never PASS.
