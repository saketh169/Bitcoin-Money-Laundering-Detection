import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from data_loader import load_elliptic_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PaperGATv2(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=4, concat=True, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
        # We concatenate original features as a skip connection; 1 output for Sigmoid binary classification
        self.classifier = nn.Linear(hidden_dim + num_features, 1)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        combined = torch.cat([h, x], dim=-1)
        # Squeeze to 1D vector of logits
        out = self.classifier(combined).squeeze(-1)
        return out

def train_paper_replica():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Using random split (70/30) as per paper methodology
    data = load_elliptic_data().to(device)
    
    # Class weights for training (pos_weight for positive/illicit class)
    train_labels = data.y[data.train_mask]
    num_licit = (train_labels == 0).sum().item()
    num_illicit = (train_labels == 1).sum().item()
    pos_weight = torch.tensor([num_licit / num_illicit], dtype=torch.float).to(device)
    
    model = PaperGATv2(num_features=data.x.size(1)).to(device)
    # BCEWithLogitsLoss applies Sigmoid numerically stable internally
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    print("Training GATv2 with SIGMOID activation...", flush=True)
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                # Apply Sigmoid to get probabilities and threshold at 0.5
                probs = torch.sigmoid(out)
                pred = (probs >= 0.5).long()
                
                # Replicate the paper's metrics (Weighted average and Accuracy)
                y_test = data.y[data.test_mask].cpu()
                p_test = pred[data.test_mask].cpu()
                
                acc = accuracy_score(y_test, p_test)
                prec = precision_score(y_test, p_test, average='weighted', zero_division=0)
                rec = recall_score(y_test, p_test, average='weighted', zero_division=0)
                f1 = f1_score(y_test, p_test, average='weighted', zero_division=0)
                
                # Also track the minority F1 for our own sanity
                minority_f1 = f1_score(y_test, p_test, pos_label=1, zero_division=0)
                
                print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | (Minority F1: {minority_f1:.4f})", flush=True)

if __name__ == "__main__":
    train_paper_replica()
