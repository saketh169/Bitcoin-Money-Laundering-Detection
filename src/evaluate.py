import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_elliptic_data
from model import GATModel
import os

def evaluate():
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data for evaluation...")
    data = load_elliptic_data().to(device)
    
    num_features = data.x.size(1)
    model = GATModel(num_features=num_features).to(device)
    model.load_state_dict(torch.load('results/best_gat_model.pth', map_location=device))
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.sigmoid(out)
        pred = (probs >= 0.5).long()
        
        test_pred = pred[data.test_mask].cpu().numpy()
        test_probs = probs[data.test_mask].cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
        
        f1 = f1_score(test_true, test_pred, pos_label=1, zero_division=0)
        prec = precision_score(test_true, test_pred, pos_label=1, zero_division=0)
        rec = recall_score(test_true, test_pred, pos_label=1, zero_division=0)
        try:
            auc = roc_auc_score(test_true, test_probs)
        except ValueError:
            auc = float('nan')
            
        print("\n--- Test Set Results (GAT) ---")
        print(f"Precision (Illicit): {prec:.4f}")
        print(f"Recall (Illicit):    {rec:.4f}")
        print(f"F1-Score (Illicit):  {f1:.4f}")
        print(f"ROC-AUC:             {auc:.4f}")
        
        cm = confusion_matrix(test_true, test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Licit (0)', 'Illicit (1)'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('GAT Confusion Matrix (Test Set)')
        plt.savefig('results/gat_confusion_matrix.png')
        print("Saved confusion matrix to results/gat_confusion_matrix.png")
        
        with open('results/gat_metrics.txt', 'w') as f:
            f.write("--- Test Set Results (GAT) ---\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"ROC-AUC: {auc:.4f}\n")

if __name__ == "__main__":
    evaluate()
