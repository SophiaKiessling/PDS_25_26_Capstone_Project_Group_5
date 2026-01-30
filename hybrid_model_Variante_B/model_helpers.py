"""
Helper Functions für Hybrid Model Prediction (Variante B)
Generiert automatisch beim Training
"""

import pandas as pd
import numpy as np
from datetime import datetime

def normalize(text):
    """Text-Normalisierung: Umlaute, Gendering, Sonderzeichen"""
    import re
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).lower()
    text = text.replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss")
    text = re.sub(r"(innen|in)\b", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_with_confidence(model, text):
    """TF-IDF Prediction mit Confidence Score"""
    text_norm = normalize(text)
    proba = model.predict_proba([text_norm])[0]
    pred_idx = int(np.argmax(proba))
    pred_label = str(model.classes_[pred_idx])
    confidence = float(proba[pred_idx])
    return pred_label, confidence

def calculate_months_between(start_date, end_date):
    """Berechnet Monate zwischen zwei Daten"""
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else pd.to_datetime(datetime.now())
        return float((end - start).days / 30)
    except:
        return 0.0

def extract_job_history_features(person_jobs, target_job_idx=0):
    """Extrahiert 24 Features aus Job-Historie"""
    target_job = person_jobs[target_job_idx]

    features = {
        'total_jobs': len(person_jobs),
        'job_number': target_job_idx + 1,
        'previous_seniority_junior': 0,
        'previous_seniority_professional': 0,
        'previous_seniority_senior': 0,
        'previous_seniority_lead': 0,
        'previous_seniority_management': 0,
        'previous_seniority_director': 0,
        'previous_dept_administrative': 0,
        'previous_dept_business_dev': 0,
        'previous_dept_consulting': 0,
        'previous_dept_customer_support': 0,
        'previous_dept_hr': 0,
        'previous_dept_it': 0,
        'previous_dept_marketing': 0,
        'previous_dept_other': 0,
        'previous_dept_project_mgmt': 0,
        'previous_dept_purchasing': 0,
        'previous_dept_sales': 0,
        'same_department_as_previous': 0,
        'months_in_current_job': 0.0,
        'avg_job_duration': 0.0,
        'seniority_increases': 0,
        'department_changes': 0,
    }

    # Previous Job Features
    if len(person_jobs) > target_job_idx + 1:
        prev_job = person_jobs[target_job_idx + 1]
        prev_sen = prev_job.get('seniority')

        if prev_sen == 'Junior': features['previous_seniority_junior'] = 1
        elif prev_sen == 'Professional': features['previous_seniority_professional'] = 1
        elif prev_sen == 'Senior': features['previous_seniority_senior'] = 1
        elif prev_sen == 'Lead': features['previous_seniority_lead'] = 1
        elif prev_sen == 'Management': features['previous_seniority_management'] = 1
        elif prev_sen == 'Director': features['previous_seniority_director'] = 1

        prev_dept = prev_job.get('department')
        dept_map = {
            'Administrative': 'previous_dept_administrative',
            'Business Development': 'previous_dept_business_dev',
            'Consulting': 'previous_dept_consulting',
            'Customer Support': 'previous_dept_customer_support',
            'Human Resources': 'previous_dept_hr',
            'Information Technology': 'previous_dept_it',
            'Marketing': 'previous_dept_marketing',
            'Other': 'previous_dept_other',
            'Project Management': 'previous_dept_project_mgmt',
            'Purchasing': 'previous_dept_purchasing',
            'Sales': 'previous_dept_sales',
        }
        if prev_dept in dept_map:
            features[dept_map[prev_dept]] = 1

        if prev_dept and target_job.get('department') and prev_dept == target_job.get('department'):
            features['same_department_as_previous'] = 1

    # Time Features
    features['months_in_current_job'] = calculate_months_between(
        target_job.get('startDate'), target_job.get('endDate')
    )

    durations = []
    for job in person_jobs:
        dur = calculate_months_between(job.get('startDate'), job.get('endDate'))
        if dur > 0:
            durations.append(dur)
    if durations:
        features['avg_job_duration'] = float(np.mean(durations))

    # Progression Features
    seniority_order = {
        'Junior': 1, 'Professional': 2, 'Senior': 3, 
        'Lead': 4, 'Management': 5, 'Director': 6
    }
    for i in range(len(person_jobs) - 1, 0, -1):
        older_sen = person_jobs[i].get('seniority')
        newer_sen = person_jobs[i-1].get('seniority')
        if older_sen and newer_sen:
            if seniority_order.get(newer_sen, 0) > seniority_order.get(older_sen, 0):
                features['seniority_increases'] += 1

    for i in range(len(person_jobs) - 1):
        if person_jobs[i].get('department') != person_jobs[i+1].get('department'):
            features['department_changes'] += 1

    return features
