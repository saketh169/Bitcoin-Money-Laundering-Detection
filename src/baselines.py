import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from data_loader import load_elliptic_data
import os

def evaluate_tabular_baselines():
    os.makedirs('results', exist_ok=True)
    print("Loading data for tabular baselines...")
    data = load_elliptic_data()
    
    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()
    
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    rf_f1 = f1_score(y_test, rf_pred, pos_label=1, zero_division=0)
    rf_prec = precision_score(y_test, rf_pred, pos_label=1, zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, pos_label=1, zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_probs)
    
    print("--- Test Set Results (Random Forest) ---")
    print(f"Precision (Illicit): {rf_prec:.4f}")
    print(f"Recall (Illicit):    {rf_rec:.4f}")
    print(f"F1-Score (Illicit):  {rf_f1:.4f}")
    print(f"ROC-AUC:             {rf_auc:.4f}")
    
    print("\nTraining MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    
    mlp_pred = mlp.predict(X_test)
    mlp_probs = mlp.predict_proba(X_test)[:, 1]
    
    mlp_f1 = f1_score(y_test, mlp_pred, pos_label=1, zero_division=0)
    mlp_prec = precision_score(y_test, mlp_pred, pos_label=1, zero_division=0)
    mlp_rec = recall_score(y_test, mlp_pred, pos_label=1, zero_division=0)
    mlp_auc = roc_auc_score(y_test, mlp_probs)
    
    print("--- Test Set Results (MLP) ---")
    print(f"Precision (Illicit): {mlp_prec:.4f}")
    print(f"Recall (Illicit):    {mlp_rec:.4f}")
    print(f"F1-Score (Illicit):  {mlp_f1:.4f}")
    print(f"ROC-AUC:             {mlp_auc:.4f}")
    
    with open('results/tabular_baselines_metrics.txt', 'w') as f:
        f.write("--- Random Forest ---\n")
        f.write(f"Precision: {rf_prec:.4f}\nRecall: {rf_rec:.4f}\nF1-Score: {rf_f1:.4f}\nROC-AUC: {rf_auc:.4f}\n\n")
        f.write("--- MLP ---\n")
        f.write(f"Precision: {mlp_prec:.4f}\nRecall: {mlp_rec:.4f}\nF1-Score: {mlp_f1:.4f}\nROC-AUC: {mlp_auc:.4f}\n")

if __name__ == "__main__":
    evaluate_tabular_baselines()
