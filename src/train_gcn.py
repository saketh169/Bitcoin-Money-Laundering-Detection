import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from data_loader import load_elliptic_data
from model import GCNModel, FocalLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def train_gcn(epochs=150, lr=0.005):
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    data = load_elliptic_data().to(device)
    
    num_features = data.x.size(1)
    model = GCNModel(num_features=num_features).to(device)
    
    # Positive class weighting for imbalanced binary BCE
    train_labels = data.y[data.train_mask]
    num_licit = (train_labels == 0).sum().item()
    num_illicit = (train_labels == 1).sum().item()
    pos_weight = torch.tensor([num_licit / num_illicit], dtype=torch.float).to(device)
    
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_f1 = 0.0
    best_model_state = None
    
    print("Starting GCN training with SIGMOID...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out_eval = model(data.x, data.edge_index)
                probs = torch.sigmoid(out_eval)
                pred = (probs >= 0.5).long()
                
                val_pred = pred[data.val_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                
                val_f1 = f1_score(val_true, val_pred, average='weighted', zero_division=0)
                val_acc = accuracy_score(val_true, val_pred)
                
                print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Val Weighted F1: {val_f1:.4f}")
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    
    print(f"Training completed. Best Val Weighted F1: {best_val_f1:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, 'results/best_gcn_model.pth')
        print("Saved best GCN model to results/best_gcn_model.pth")
    
    # Test Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.sigmoid(out).cpu().numpy()
        pred = (probs >= 0.5).astype(int)
        
        test_mask = data.test_mask.cpu().numpy()
        test_pred = pred[test_mask]
        test_probs = probs[test_mask]
        test_true = data.y.cpu().numpy()[test_mask]
        
        # Combined / Weighted metrics
        acc = accuracy_score(test_true, test_pred)
        prec_weighted = precision_score(test_true, test_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(test_true, test_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(test_true, test_pred, average='weighted', zero_division=0)
        
        # Illicit-only metrics
        prec_illicit = precision_score(test_true, test_pred, pos_label=1, zero_division=0)
        rec_illicit = recall_score(test_true, test_pred, pos_label=1, zero_division=0)
        f1_illicit = f1_score(test_true, test_pred, pos_label=1, zero_division=0)
        
        try:
            auc = roc_auc_score(test_true, test_probs)
        except ValueError:
            auc = float('nan')
            
        print("\n================ TEST SET RESULTS (GCN) ================")
        print(f"Overall Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
        print(f"Overall Weighted Prec:  {prec_weighted:.4f}")
        print(f"Overall Weighted Rec:   {rec_weighted:.4f}")
        print(f"Overall Weighted F1:    {f1_weighted:.4f}")
        print(f"Illicit-only Prec:      {prec_illicit:.4f}")
        print(f"Illicit-only Rec:       {rec_illicit:.4f}")
        print(f"Illicit-only F1:        {f1_illicit:.4f}")
        print(f"ROC-AUC:                {auc:.4f}")
        print("========================================================\n")
        
        # Save metrics file
        with open('results/gcn_metrics.txt', 'w') as f:
            f.write("--- Test Set Results (GCN Baseline) ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Weighted Precision: {prec_weighted:.4f}\n")
            f.write(f"Weighted Recall: {rec_weighted:.4f}\n")
            f.write(f"Weighted F1: {f1_weighted:.4f}\n")
            f.write(f"Illicit Precision: {prec_illicit:.4f}\n")
            f.write(f"Illicit Recall: {rec_illicit:.4f}\n")
            f.write(f"Illicit F1: {f1_illicit:.4f}\n")
            f.write(f"ROC-AUC: {auc:.4f}\n")
            
        # Plot confusion matrix
        cm = confusion_matrix(test_true, test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Licit (0)', 'Illicit (1)'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('GCN Confusion Matrix (Test Set)')
        plt.savefig('results/gcn_confusion_matrix.png')
        plt.close()
        print("Saved confusion matrix to results/gcn_confusion_matrix.png")

if __name__ == "__main__":
    train_gcn()
