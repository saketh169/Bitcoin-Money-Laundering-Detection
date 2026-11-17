import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from data_loader import load_elliptic_data
from sklearn.metrics import f1_score

class HybridGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout_rate=0.3):
        super(HybridGAT, self).__init__()
        # GATv2 layers to extract graph topology
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=4, concat=True, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
        # Classifier that looks at BOTH graph features AND raw tabular features
        self.classifier = nn.Linear(hidden_dim + num_features, num_classes)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        # SKIP CONNECTION: Concatenate GAT features (h) with original features (x)
        combined = torch.cat([h, x], dim=-1)
        
        out = self.classifier(combined)
        return out

def train_hybrid_gat():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_elliptic_data().to(device)
    
    # Calculate class weights for cross entropy
    train_labels = data.y[data.train_mask]
    num_licit = (train_labels == 0).sum().item()
    num_illicit = (train_labels == 1).sum().item()
    total = num_licit + num_illicit
    weight_licit = total / (2.0 * num_licit)
    weight_illicit = total / (2.0 * num_illicit)
    class_weights = torch.tensor([weight_licit, weight_illicit], dtype=torch.float).to(device)
    
    model = HybridGAT(num_features=data.x.size(1)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    best_val_f1 = 0
    best_test_f1 = 0
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                
                val_f1 = f1_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu(), pos_label=1, zero_division=0)
                test_f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), pos_label=1, zero_division=0)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_test_f1 = test_f1
                print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}", flush=True)

if __name__ == "__main__":
    train_hybrid_gat()
