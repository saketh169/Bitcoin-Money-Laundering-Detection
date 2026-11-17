import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=1, dropout_rate=0.3):
        super(GATModel, self).__init__()
        # Layer 1: 4 attention heads (K=4). Output dim is hidden_dim * 4 due to concatenation
        self.conv1 = GATConv(num_features, hidden_dim, heads=4, concat=True, dropout=dropout_rate)
        
        # Layer 2: 1 attention head. 
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
        # Binary classification head (1 output for Sigmoid)
        self.classifier = nn.Linear(hidden_dim, 1)
        
        self.dropout_rate = dropout_rate
        
    def forward(self, x, edge_index):
        # Layer 1 -> ELU -> Dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Layer 2 -> ELU -> Dropout
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Linear head producing single logit per node
        out = self.classifier(x).squeeze(-1)
        return out

class GCNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=1, dropout_rate=0.3):
        super(GCNModel, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        out = self.classifier(x).squeeze(-1)
        return out

# Binary Focal Loss with Sigmoid
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits: [N] raw unnormalized scores
        # targets: [N] binary 0/1 labels
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none', pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        p_t = p * targets.float() + (1.0 - p) * (1.0 - targets.float())
        focal_loss = ((1.0 - p_t) ** self.gamma) * bce
        return focal_loss.mean()
