# PDS_25_26_Capstone_Project_Group_5
Capstone Project of Group 5

## Group Members
- Quynh Anh Ha
- Viktoria Rupp
- Sophia Kießling

## Exploratory Data Analysis via Streamlit
### PDS_Dashboard.py
Purpose: Code for the exploratory data analysis and presentation with streamlit

## Notebooks with applied approaches

### 01_Rule_based_labeling.ipynb
Purpose: Implementation of the Rule-based matching (baseline)

### 02_Embedding_based_labeling.ipynb
Purpose: Implementation of Embedding-based Labeling with MiniLM-L12 and DistilUSE

### 03_Fine_tuned_classification_model.ipynb
Purpose: Implementation of Fine-tuned classification models (xlm-roberta-base and distilbert-base-multilingual-cased)  
Models: the models are available here:   
        https://huggingface.co/SophiaKiessling/PDS_2025_Capstone_Project_Group5_best_dept_model
        https://huggingface.co/SophiaKiessling/PDS_2025_Capstone_Project_Group5_best_sen_model  
Related file: label_encoders.pkl
Note: Notebook was created and testet in Google Colab due to GPU requirements  
      Notebook outputs were cleared to avoid GitHub rendering issues related to Colab-specific metadata

### 04_Programmatic_labeling_approach.ipynb
Purpose: Implementation of programmatic labeling to automatically generate pseudo-labels

### 06_Simple_interpretable_baseline.ipynb
Purpose: Implementation of a simple and interpretable supervised baseline using TF-IDF features and Logistic Regression

### 07_Hybrid_Model_and_Pseudolabeling.ipynb
Purpose: Implementation of Pseudolabeling (ensemble with TF-IDF + LogReg, Rule-Based, Fine-Tuning), Feature Engineering and Random Forest training, Hybrid prediction with TF-IDF + LogReg and Random Forest
Model: hybrid_model  
Related file: linkedin_pseudo_labeled_ensemble.json

### 07_Hybrid_Model_and_Pseudolabeling_Variante_B.ipynb
Purpose: variant of the hybrid prediction model with AI generated pseudolabels for "professional"  
Model: hybrid_model_Variante_B  
Related file: linkedin_pseudo_labeled_ensemble_professional.json

## Given Datasets

### department-v2.csv

### seniority-v2.csv

### linkedin-cvs.annotate.json

### linkedin-cvs-not-annotated.json




