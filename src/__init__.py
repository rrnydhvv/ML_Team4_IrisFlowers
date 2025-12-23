import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like 'src.models' work
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from models.KNN import knn_predict_weighted, evaluate_knn_kfold, run_knn_train_test, save_model_KNN, load_model_KNN

