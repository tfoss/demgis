"""
Batch process African countries with 2km DEM resolution.

This version uses 2km DEM and doubles XY_MM_PER_PIXEL to maintain
the same physical print size as 1km version.
"""

import sys
import os

# Import everything from make_africa
sys.path.insert(0, os.path.dirname(__file__))
from make_africa import *

# Override XY_MM_PER_PIXEL to compensate for 2x pixel size
# This ensures the same physical dimensions as 1km version
import make_all_sa_with_vector_clip as sa_module
sa_module.XY_MM_PER_PIXEL = 0.50  # Double the standard 0.25 for 2km pixels

# Re-import the module-level constant into local scope
XY_MM_PER_PIXEL = 0.50

if __name__ == "__main__":
    main()
