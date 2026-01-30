"""
Hybrid Predictor Class für Streamlit Deployment
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import sys
import os
from typing import Dict, List, Any

# Fix imports - add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Now import helpers
from model_helpers import normalize, predict_with_confidence, extract_job_history_features

class HybridPredictor:
    """
    Hybrid Model: TF-IDF + Random Forest
    """

    def __init__(self, model_dir='hybrid_model'):
        """Lädt alle Modell-Komponenten"""

        # Make paths absolute
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(os.getcwd(), model_dir)

        print(f"Lade Modelle aus: {model_dir}")

        # Load TF-IDF Models
        self.tfidf_sen = joblib.load(f'{model_dir}/tfidf_seniority.pkl')
        self.tfidf_dept = joblib.load(f'{model_dir}/tfidf_department.pkl')

        # Load Random Forest Models
        self.rf_sen = joblib.load(f'{model_dir}/rf_seniority.pkl')
        self.rf_dept = joblib.load(f'{model_dir}/rf_department.pkl')

        # Load Label Encoders
        self.le_sen_rf = joblib.load(f'{model_dir}/le_seniority_rf.pkl')
        self.le_dept_rf = joblib.load(f'{model_dir}/le_department_rf.pkl')

        # Load Config
        with open(f'{model_dir}/config.pkl', 'rb') as f:
            config = pickle.load(f)
            self.hybrid_cfg = config['hybrid_config']
            self.feature_cols = config['feature_cols']

        # Expected Features
        self.rf_sen_features = list(self.rf_sen.feature_names_in_)
        self.rf_dept_features = list(self.rf_dept.feature_names_in_)

        print(f" Modelle geladen")
        print(f"   Seniority Classes: {list(self.le_sen_rf.classes_)}")
        print(f"   Department Classes: {list(self.le_dept_rf.classes_)}")

    def predict_seniority(self, person_jobs: List[Dict], target_job_idx: int = 0) -> Dict[str, Any]:
        """Predict Seniority für einen Job"""
        job = person_jobs[target_job_idx]
        text = str(job.get("position", "")).strip()

        # TF-IDF Prediction
        tfidf_pred, tfidf_conf = predict_with_confidence(self.tfidf_sen, text)

        # Random Forest Prediction
        features = extract_job_history_features(person_jobs, target_job_idx)
        feature_df = pd.DataFrame([features])

        # Ensure all required features exist
        for feat in self.rf_sen_features:
            if feat not in feature_df.columns:
                feature_df[feat] = 0

        feature_vector = feature_df[self.rf_sen_features].fillna(0)
        rf_pred_idx = self.rf_sen.predict(feature_vector)[0]
        rf_probs = self.rf_sen.predict_proba(feature_vector)[0]
        rf_conf = rf_probs.max()
        rf_pred = self.le_sen_rf.inverse_transform([rf_pred_idx])[0]

        # Combination Logic
        if tfidf_conf >= self.hybrid_cfg['base_hi']:
            return {'label': tfidf_pred, 'confidence': tfidf_conf, 'source': 'tfidf'}
        if rf_conf >= self.hybrid_cfg['rf_hi']:
            return {'label': rf_pred, 'confidence': rf_conf, 'source': 'rf'}

        # Else: Higher confidence wins
        if tfidf_conf >= rf_conf:
            return {'label': tfidf_pred, 'confidence': tfidf_conf, 'source': 'tfidf_fallback'}
        else:
            return {'label': rf_pred, 'confidence': rf_conf, 'source': 'rf_fallback'}

    def predict_department(self, person_jobs: List[Dict], target_job_idx: int = 0) -> Dict[str, Any]:
        """Predict Department für einen Job"""
        job = person_jobs[target_job_idx]
        text = str(job.get("position", "")).strip()

        # TF-IDF Prediction
        tfidf_pred, tfidf_conf = predict_with_confidence(self.tfidf_dept, text)

        # Random Forest Prediction
        features = extract_job_history_features(person_jobs, target_job_idx)
        feature_df = pd.DataFrame([features])

        # Ensure all required features exist
        for feat in self.rf_dept_features:
            if feat not in feature_df.columns:
                feature_df[feat] = 0

        feature_vector = feature_df[self.rf_dept_features].fillna(0)
        rf_pred_idx = self.rf_dept.predict(feature_vector)[0]
        rf_probs = self.rf_dept.predict_proba(feature_vector)[0]
        rf_conf = rf_probs.max()
        rf_pred = self.le_dept_rf.inverse_transform([rf_pred_idx])[0]

        # Combination Logic
        if tfidf_conf >= self.hybrid_cfg['base_hi']:
            return {'label': tfidf_pred, 'confidence': tfidf_conf, 'source': 'tfidf'}
        if rf_conf >= self.hybrid_cfg['rf_hi']:
            return {'label': rf_pred, 'confidence': rf_conf, 'source': 'rf'}

        # Department Fallback: Use previous job if both confidences are low
        if self.hybrid_cfg['dept_fallback'] and tfidf_conf < 0.6 and rf_conf < 0.6:
            if len(person_jobs) > target_job_idx + 1:
                prev_dept = person_jobs[target_job_idx + 1].get("department")
                if prev_dept:
                    return {'label': prev_dept, 'confidence': 0.5, 'source': 'previous_job'}

        # Else: Higher confidence wins
        if tfidf_conf >= rf_conf:
            return {'label': tfidf_pred, 'confidence': tfidf_conf, 'source': 'tfidf_fallback'}
        else:
            return {'label': rf_pred, 'confidence': rf_conf, 'source': 'rf_fallback'}

    def predict(self, person_jobs: List[Dict], target_job_idx: int = 0) -> Dict[str, Any]:
        """Predict beide Tasks für einen Job"""
        sen_result = self.predict_seniority(person_jobs, target_job_idx)
        dept_result = self.predict_department(person_jobs, target_job_idx)

        return {
            'seniority': sen_result,
            'department': dept_result
        }
