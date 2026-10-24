import os
import sys

import matplotlib

matplotlib.use("Agg")

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in [REPO_DIR, os.path.join(REPO_DIR, "modules")]:
    if path not in sys.path:
        sys.path.insert(0, path)
