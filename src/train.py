import torch
import numpy as np
import os
from data_loader import load_elliptic_data
from model import GATModel, FocalLoss
from sklearn.metrics import f1_score, precision_score, recall_score

def train_model(epochs=200, lr=0.001):
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    data = load_elliptic_data().to(device)
    
    num_features = data.x.size(1)
    model = GATModel(num_features=num_features).to(device)
    
    # Positive class weighting for imbalanced binary BCE
    train_labels = data.y[data.train_mask]
    num_licit = (train_labels == 0).sum().item()
    num_illicit = (train_labels == 1).sum().item()
    pos_weight = torch.tensor([num_licit / num_illicit], dtype=torch.float).to(device)
    
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0.0
    best_model_state = None
    
    print("Starting training with SIGMOID...")
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
                
                # Validation metrics
                val_pred = pred[data.val_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                
                val_f1 = f1_score(val_true, val_pred, pos_label=1, zero_division=0)
                val_prec = precision_score(val_true, val_pred, pos_label=1, zero_division=0)
                val_rec = recall_score(val_true, val_pred, pos_label=1, zero_division=0)
                
                print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | "
                      f"Val F1 (Illicit): {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    
    print(f"Training completed. Best Val F1: {best_val_f1:.4f}")
    
    if best_model_state is not None:
        torch.save(best_model_state, 'results/best_gat_model.pth')
        print("Saved best model to results/best_gat_model.pth")
    else:
        # Fallback if F1 was 0 all the way
        torch.save(model.state_dict(), 'results/best_gat_model.pth')

if __name__ == "__main__":
    train_model(epochs=150)
