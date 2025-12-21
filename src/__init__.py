import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like 'src.models' work
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.models.KNN import run_knn_train_test, fit_knn, predict_single, predict_batch

# Expose KNN models at project root level
__all__ = ["run_knn_train_test", "fit_knn", "predict_single", "predict_batch", "project_root"]
