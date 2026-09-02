"""TMJ Condyle 3D data and experiment utilities.

The project intentionally contains only one foreground class:
1 = mandibular_condyle.
"""

from .config import (
    CONDYLE_LABEL,
    DATASET_ID,
    DATASET_NAME,
    PROJECT_ROOT,
)

__all__ = [
    "CONDYLE_LABEL",
    "DATASET_ID",
    "DATASET_NAME",
    "PROJECT_ROOT",
]
