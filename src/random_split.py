"""
========================================================================================
Bitcoin Money Laundering Detection Using Graph Attention Networks (GAT)
Random Split Pipeline & Model Evaluation (random_split.py)

Course: Advanced Deep Learning (ADL)
Authors: Saketh Pabbu & Nerella Venkata Sriram
Institution: Indian Institute of Information Technology, Sri City (IIIT Sri City)
========================================================================================
Description:
1. Generates stratified 70% / 15% / 15% (Train / Val / Test) split on the 46,564 labeled nodes
   in the Elliptic Bitcoin Dataset (random_state=42, zero data leakage).
2. Fits z-score normalization strictly on the training partition.
3. Saves split masks & indices to 'data/random_split_indices.pt'.
4. Implements training and testing for PaperGATv2 (with skip connection and pos_weight=9.245).
5. Supports baseline comparisons (Random Forest, MLP, GCN) and generates ROC & CM plots.
========================================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure local imports work whether run from project root or src/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from torch_geometric.datasets import EllipticBitcoinDataset
    from torch_geometric.nn import GATv2Conv, GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ======================================================================================
# 1. ARCHITECTURE DEFINITIONS
# ======================================================================================
if HAS_PYG:
    class PaperGATv2(nn.Module):
        """
        Dynamic GATv2 architecture with:
        - Layer 1: GATv2Conv with K=4 attention heads (hidden_dim=128)
        - Layer 2: GATv2Conv with K=1 head (hidden_dim=128)
        - Skip Connection: Concatenation of 128-D graph embedding + 165-D raw features = 293-D
        - Classifier: Linear(293 -> 1) with Sigmoid activation
        """
        def __init__(self, num_features=165, hidden_dim=128, dropout_rate=0.3):
            super().__init__()
            self.conv1 = GATv2Conv(num_features, hidden_dim, heads=4, concat=True, dropout=dropout_rate)
            self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
            self.classifier = nn.Linear(hidden_dim + num_features, 1)
            self.dropout_rate = dropout_rate

        def forward(self, x, edge_index):
            # Layer 1
            h = self.conv1(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            
            # Layer 2
            h = self.conv2(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            
            # Skip connection: [graph embedding || raw features]
            combined = torch.cat([h, x], dim=-1)
            out = self.classifier(combined).squeeze(-1)
            return out


    class GCNBaseline(nn.Module):
        def __init__(self, num_features=165, hidden_dim=64, dropout_rate=0.3):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, 1)
            self.dropout_rate = dropout_rate

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = self.conv2(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            out = self.classifier(h).squeeze(-1)
            return out


# ======================================================================================
# 2. DATASET LOADING & STRATIFIED RANDOM SPLIT (70 / 15 / 15)
# ======================================================================================
def get_random_split_data(root='data/EllipticBitcoinDataset', random_state=42, save_split=True):
    """
    Loads EllipticBitcoinDataset and produces the exact 70/15/15 stratified random split
    on labeled nodes (Class 0: Licit, Class 1: Illicit).
    Ensures zero leakage by calculating z-score standardization exclusively on train_x.
    """
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required to load EllipticBitcoinDataset.")

    # Locate raw data directory
    raw_dir = os.path.join(root, 'raw')
    req_files = ['elliptic_txs_features.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_classes.csv']
    missing_files = [f for f in req_files if not os.path.exists(os.path.join(raw_dir, f))]

    if missing_files:
        print("=" * 80)
        print("WARNING: Required raw dataset files are missing in:")
        print(f"  --> {os.path.abspath(raw_dir)}")
        print(f"Missing: {missing_files}")
        print("\nPlease download the files from:")
        print("  1. https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip")
        print("  2. https://data.pyg.org/datasets/elliptic/elliptic_txs_edgelist.csv.zip")
        print("  3. https://data.pyg.org/datasets/elliptic/elliptic_txs_classes.csv.zip")
        print("Extract the 3 CSV files directly into the above folder.")
        print("=" * 80)
        raise FileNotFoundError(f"Missing dataset files in {raw_dir}")

    print(f"Loading Elliptic dataset from: {os.path.abspath(root)} ...")
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    # Filter labeled nodes only (0: Licit, 1: Illicit, 2: Unknown)
    labeled_mask = (data.y == 0) | (data.y == 1)
    labeled_indices = labeled_mask.nonzero(as_tuple=False).view(-1).numpy()
    labeled_labels = data.y[labeled_mask].numpy()

    total_labeled = len(labeled_indices)
    print(f"Total nodes in graph: {data.num_nodes:,}")
    print(f"Total labeled nodes:  {total_labeled:,} (Licit: {(labeled_labels == 0).sum():,}, Illicit: {(labeled_labels == 1).sum():,})")

    # Stratified 70% Train, 30% Temporary (Test + Val)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        labeled_indices,
        labeled_labels,
        test_size=0.30,
        random_state=random_state,
        stratify=labeled_labels
    )

    # Stratified 50/50 split of the 30% temp -> 15% Val, 15% Test
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    # Create boolean masks for full graph
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Zero-leakage Z-Score Normalization (fitted strictly on training partition)
    train_x = data.x[data.train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    std[std == 0] = 1.0
    data.x = (data.x - mean) / std

    print("\n--- Stratified Random Split Summary (70 / 15 / 15) ---")
    print(f"  Train Set: {train_mask.sum().item():>6} nodes (Licit: {(data.y[train_mask] == 0).sum():>5}, Illicit: {(data.y[train_mask] == 1).sum():>5})")
    print(f"  Val Set:   {val_mask.sum().item():>6} nodes (Licit: {(data.y[val_mask] == 0).sum():>5}, Illicit: {(data.y[val_mask] == 1).sum():>5})")
    print(f"  Test Set:  {test_mask.sum().item():>6} nodes (Licit: {(data.y[test_mask] == 0).sum():>5}, Illicit: {(data.y[test_mask] == 1).sum():>5})")

    if save_split:
        save_path = os.path.join(root, 'random_split_indices.pt')
        torch.save({
            'train_idx': torch.tensor(train_idx),
            'val_idx': torch.tensor(val_idx),
            'test_idx': torch.tensor(test_idx),
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }, save_path)
        print(f"Split indices and masks saved to: {os.path.abspath(save_path)}")

    return data


# ======================================================================================
# 3. TABULAR BASELINE EVALUATION (Random Forest & MLP)
# ======================================================================================
def evaluate_tabular(data):
    print("\n" + "=" * 80)
    print("Evaluating Tabular Baselines on Random Split...")
    print("=" * 80)

    X_train = data.x[data.train_mask].cpu().numpy()
    y_train = data.y[data.train_mask].cpu().numpy()
    X_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    # 1. Random Forest
    print("\n[1/2] Training Random Forest Baseline (100 trees, balanced weights)...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred, pos_label=1, zero_division=0),
        'Recall': recall_score(y_test, rf_pred, pos_label=1, zero_division=0),
        'F1': f1_score(y_test, rf_pred, pos_label=1, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, rf_probs),
        'probs': rf_probs,
        'pred': rf_pred
    }

    print(f"  Random Forest -> Accuracy: {rf_metrics['Accuracy']*100:.2f}% | Precision: {rf_metrics['Precision']*100:.2f}% | Recall: {rf_metrics['Recall']*100:.2f}% | F1: {rf_metrics['F1']*100:.2f}% | ROC-AUC: {rf_metrics['ROC_AUC']:.4f}")

    # 2. MLP Baseline
    print("\n[2/2] Training Multi-Layer Perceptron (MLP) Baseline...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_probs = mlp.predict_proba(X_test)[:, 1]

    mlp_metrics = {
        'Accuracy': accuracy_score(y_test, mlp_pred),
        'Precision': precision_score(y_test, mlp_pred, pos_label=1, zero_division=0),
        'Recall': recall_score(y_test, mlp_pred, pos_label=1, zero_division=0),
        'F1': f1_score(y_test, mlp_pred, pos_label=1, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, mlp_probs),
        'probs': mlp_probs,
        'pred': mlp_pred
    }

    print(f"  MLP           -> Accuracy: {mlp_metrics['Accuracy']*100:.2f}% | Precision: {mlp_metrics['Precision']*100:.2f}% | Recall: {mlp_metrics['Recall']*100:.2f}% | F1: {mlp_metrics['F1']*100:.2f}% | ROC-AUC: {mlp_metrics['ROC_AUC']:.4f}")

    return rf_metrics, mlp_metrics


# ======================================================================================
# 4. GATv2 TRAINING & EVALUATION
# ======================================================================================
def train_and_eval_gat(data, epochs=150, lr=0.005, weight_decay=1e-4):
    print("\n" + "=" * 80)
    print("Training PaperGATv2 Model on Random Split...")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data = data.to(device)
    num_features = data.x.size(1)

    # Imbalance pos_weight (42,019 / 4,545 approx 9.245)
    train_labels = data.y[data.train_mask]
    num_licit = (train_labels == 0).sum().item()
    num_illicit = (train_labels == 1).sum().item()
    pos_weight = torch.tensor([num_licit / num_illicit], dtype=torch.float).to(device)
    print(f"Calculated Positive Imbalance Weight (pos_weight): {pos_weight.item():.4f}")

    model = PaperGATv2(num_features=num_features, hidden_dim=128, dropout_rate=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_probs = torch.sigmoid(val_out)
                val_pred = (val_probs >= 0.5).long()

                y_val = data.y[data.val_mask].cpu().numpy()
                p_val = val_pred[data.val_mask].cpu().numpy()

                val_f1 = f1_score(y_val, p_val, pos_label=1, zero_division=0)
                val_prec = precision_score(y_val, p_val, pos_label=1, zero_division=0)
                val_rec = recall_score(y_val, p_val, pos_label=1, zero_division=0)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f} | Val F1 (Illicit): {val_f1*100:.2f}% | Prec: {val_prec*100:.2f}% | Rec: {val_rec*100:.2f}%")

    # Load best model for test evaluation
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.sigmoid(out)
        pred = (probs >= 0.5).long()

        y_test = data.y[data.test_mask].cpu().numpy()
        p_test = pred[data.test_mask].cpu().numpy()
        prob_test = probs[data.test_mask].cpu().numpy()

        gat_metrics = {
            'Accuracy': accuracy_score(y_test, p_test),
            'Precision': precision_score(y_test, p_test, pos_label=1, zero_division=0),
            'Recall': recall_score(y_test, p_test, pos_label=1, zero_division=0),
            'F1': f1_score(y_test, p_test, pos_label=1, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, prob_test),
            'probs': prob_test,
            'pred': p_test,
            'true': y_test
        }

    os.makedirs('results', exist_ok=True)
    torch.save(best_weights or model.state_dict(), 'results/best_random_split_gat.pth')
    print(f"\nBest model checkpoint saved to: results/best_random_split_gat.pth")
    print(f"GAT Test Results -> Accuracy: {gat_metrics['Accuracy']*100:.2f}% | Precision: {gat_metrics['Precision']*100:.2f}% | Recall: {gat_metrics['Recall']*100:.2f}% | F1: {gat_metrics['F1']*100:.2f}% | ROC-AUC: {gat_metrics['ROC_AUC']:.4f}")

    return gat_metrics


# ======================================================================================
# 5. VISUALIZATION & PLOT GENERATION
# ======================================================================================
def generate_random_split_plots(y_test, gat_metrics, rf_metrics=None, mlp_metrics=None):
    os.makedirs('results', exist_ok=True)
    print("\nGenerating publication-quality plots in 'results/' directory...")

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, gat_metrics['pred'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Licit (0)', 'Illicit (1)'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('GAT Test Confusion Matrix (Random Split: 70/15/15)', fontsize=12, pad=12)
    plt.tight_layout()
    cm_path = 'results/gat_random_split_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [Saved] Confusion Matrix -> {cm_path}")

    # 2. ROC-AUC Curves Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr_gat, tpr_gat, _ = roc_curve(y_test, gat_metrics['probs'])
    ax.plot(fpr_gat, tpr_gat, color='#1f77b4', lw=2.5, label=f'GAT (AUC = {gat_metrics["ROC_AUC"]:.4f})')

    if rf_metrics is not None:
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_metrics['probs'])
        ax.plot(fpr_rf, tpr_rf, color='#2ca02c', lw=2.0, linestyle='--', label=f'Random Forest (AUC = {rf_metrics["ROC_AUC"]:.4f})')

    if mlp_metrics is not None:
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_metrics['probs'])
        ax.plot(fpr_mlp, tpr_mlp, color='#ff7f0e', lw=2.0, linestyle=':', label=f'MLP (AUC = {mlp_metrics["ROC_AUC"]:.4f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall / TPR)', fontsize=12)
    ax.set_title('ROC Curves Comparison on Held-Out Random Split Test Set', fontsize=12, pad=12)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = 'results/random_split_roc_curves.png'
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  [Saved] ROC Curves -> {roc_path}")

    # 3. Benchmark Metrics Comparison Bar Chart
    if rf_metrics is not None and mlp_metrics is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_names = ['Precision (Illicit)', 'Recall (Illicit)', 'F1-Score (Illicit)', 'ROC-AUC']
        gat_vals = [gat_metrics['Precision'], gat_metrics['Recall'], gat_metrics['F1'], gat_metrics['ROC_AUC']]
        rf_vals = [rf_metrics['Precision'], rf_metrics['Recall'], rf_metrics['F1'], rf_metrics['ROC_AUC']]
        mlp_vals = [mlp_metrics['Precision'], mlp_metrics['Recall'], mlp_metrics['F1'], mlp_metrics['ROC_AUC']]

        x = np.arange(len(metrics_names))
        width = 0.25

        ax.bar(x - width, rf_vals, width, label='Random Forest', color='#2ca02c')
        ax.bar(x, mlp_vals, width, label='MLP', color='#ff7f0e')
        ax.bar(x + width, gat_vals, width, label='GAT (Ours)', color='#1f77b4')

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Benchmark on Held-Out Test Set (Random Split)', fontsize=12, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        bar_path = 'results/random_split_model_comparison.png'
        plt.savefig(bar_path, dpi=300)
        plt.close()
        print(f"  [Saved] Model Comparison Bar Chart -> {bar_path}")


# ======================================================================================
# 6. MAIN EXECUTION ROUTINE
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Bitcoin GAT Random Split Pipeline")
    parser.add_argument('--root', type=str, default='data/EllipticBitcoinDataset', help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=150, help='Number of GAT training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--split_only', action='store_true', help='Only generate and save the split masks')
    args = parser.parse_args()

    # Step 1: Generate or load data with stratified random split
    data = get_random_split_data(root=args.root, random_state=42, save_split=True)

    if args.split_only:
        print("\nSplit successfully created and saved. Exiting as --split_only was specified.")
        return

    # Step 2: Evaluate tabular baselines
    rf_metrics, mlp_metrics = evaluate_tabular(data)

    # Step 3: Train and evaluate dynamic GATv2
    gat_metrics = train_and_eval_gat(data, epochs=args.epochs, lr=args.lr)

    # Step 4: Generate visual plots
    y_test = data.y[data.test_mask].cpu().numpy()
    generate_random_split_plots(y_test, gat_metrics, rf_metrics, mlp_metrics)

    print("\n" + "=" * 80)
    print("ALL RANDOM SPLIT OPERATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
