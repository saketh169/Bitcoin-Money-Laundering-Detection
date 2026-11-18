# Elliptic Bitcoin Dataset: Architecture, Features & Evaluation Guide

**Paper Reference:** *"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"* (Weber et al., ACM SIGKDD 2019, MIT-IBM Watson AI Lab)  
**Project:** Bitcoin Money Laundering Detection Using Graph Attention Networks (GAT)  
**Authors:** Saketh Pabbu & Nerella Venkata Sriram  
**Institution:** Indian Institute of Information Technology, Sri City (IIIT Sri City)  

---

## 1. Overview & Download Links

The **Elliptic Bitcoin Dataset** is the premier open-source benchmark for financial graph deep learning, mapping real transaction flows on the public Bitcoin blockchain to detect illicit entities.

### Official Download Links:
* **Direct PyTorch Geometric (PyG) Downloads (Fastest):**
  * Features CSV: [https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip](https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip) (~143 MB)
  * Edgelist CSV: [https://data.pyg.org/datasets/elliptic/elliptic_txs_edgelist.csv.zip](https://data.pyg.org/datasets/elliptic/elliptic_txs_edgelist.csv.zip) (~1.6 MB)
  * Classes CSV: [https://data.pyg.org/datasets/elliptic/elliptic_txs_classes.csv.zip](https://data.pyg.org/datasets/elliptic/elliptic_txs_classes.csv.zip) (~0.9 MB)
* **Kaggle Mirror:** [Elliptic Data Set on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

### Expected Directory Layout:
Extract the 3 uncompressed CSV files into:
```
Bitcoin_GAT_Project_Package/
└── data/
    └── EllipticBitcoinDataset/
        └── raw/
            ├── elliptic_txs_features.csv
            ├── elliptic_txs_edgelist.csv
            └── elliptic_txs_classes.csv
```

---

## 2. Graph Topography & Composition

```
  [203,769 Transactions (Nodes)]  ──► Entities/transactions in the financial network.
  [234,355 Payment Links (Edges)] ──► Directed Bitcoin flows from input to output.
  [165 Features per Node]         ──► Quantitative financial properties.
  [49 Time Steps]                 ──► Spanning ~2-week temporal windows.
```

* **`203,769` Graph Nodes ($\mathcal{V}$):** Individual Bitcoin transactions executed on the public blockchain.
* **`234,355` Directed Edges ($\mathcal{E}$):** A directed link $(j \to i)$ indicates that transaction $j$ provided input Bitcoins that were consumed and spent by transaction $i$.
* **`49` Discrete Time Steps:** Each timestep represents an interval of roughly 2 weeks. All transactions in a timestep appear simultaneously and only send funds to transactions in the same or future timesteps.

---

## 3. Node Class Breakdown & Ground Truth

Not all Bitcoin transactions can be definitively linked to real-world identities. The dataset is partitioned into three distinct classes:

| Class Label | Category | Count | Percentage | Definition & Forensic Ground Truth | Role in Model Training |
|:---:|:---|:---:|:---:|:---|:---|
| **0** | **Licit** | **42,019** | 20.6% | Verified legal transactions: regulated exchanges, miners, legal merchants, payroll wallets. | Supervised Negative ($y=0$) |
| **1** | **Illicit** | **4,545** | 2.2% | Verified criminal transactions: darknet markets, ransomware cartels (e.g., WannaCry), scams, terrorist financing. | Supervised Positive ($y=1$) |
| **2** | **Unknown** | **157,205** | 77.2% | Public transactions with no verified forensic ground truth label. | Retained in graph for message passing; masked from loss & evaluation. |
| **Total** | **All Nodes** | **203,769** | 100.0% | Complete transaction graph | Full Network |

---

## 4. Why We Combine Licit and Illicit for Accuracy & Evaluation

A central design choice in our training and evaluation pipeline is filtering for only the **labeled transactions**:

$$\text{Labeled Subgraph} = \mathcal{V}_{\text{labeled}} = \{ v_i \in \mathcal{V} \mid y_i \in \{0, 1\} \}$$
$$|\mathcal{V}_{\text{labeled}}| = 42{,}019 + 4{,}545 = \mathbf{46{,}564} \text{ nodes}$$

### 1. Why Unknown Transactions Must Be Masked from Accuracy
* Evaluating a classification model requires a known ground-truth target.
* Because 77.2% (157,205) of the nodes are **Unknown**, computing accuracy over the full 203,769 nodes would either treat unknown nodes as normal (which is factually false) or fabricate arbitrary labels.
* Therefore, accuracy and all evaluation metrics (Precision, Recall, F1-Score, ROC-AUC) are calculated strictly over the combined **Licit ($y=0$) and Illicit ($y=1$) nodes**.

### 2. Why Unknown Nodes Must Still Remain in the Graph
* While unknown nodes are masked out of the loss function ($\mathcal{L}$) and accuracy calculation, they **cannot be removed from the graph structure**.
* Criminal money launderers use peeling chains and multi-hop routing through intermediary addresses. These intermediate hops are often labeled "Unknown".
* If unknown nodes were deleted, the transaction graph would shatter into thousands of disconnected singletons, completely crippling the Graph Attention Network's ability to pass messages along payment chains.
* Our `PaperGATv2` model allows unknown nodes to participate in message passing ($\mathbf{H}^{(l+1)} = \text{GAT}(\mathbf{H}^{(l)}, \mathcal{E})$), but loss and gradients are computed solely across `data.train_mask & (y == 0 | y == 1)`.

### 3. How Accuracy Combines Licit and Illicit
Accuracy measures the model's total correctness across **both** legal commerce and money laundering:

$$\text{Accuracy} = \frac{\text{True Negatives (Correct Licit)} + \text{True Positives (Correct Illicit)}}{\text{Total Labeled Evaluation Nodes}}$$

On our held-out test set ($N_{\text{test}} = 6{,}984$ nodes):
* **True Negatives (TN):** $6{,}014$ licit transactions correctly identified.
* **True Positives (TP):** $525$ illicit transactions correctly intercepted.
* **False Positives (FP):** $288$ licit transactions falsely flagged.
* **False Negatives (FN):** $157$ illicit transactions missed.

$$\text{Accuracy} = \frac{6{,}014 + 525}{6{,}984} = \frac{6{,}539}{6{,}984} = \mathbf{93.54\%}$$

---

## 5. Feature Architecture (165 Continuous Numerical Features)

Each node $v_i$ contains a feature vector $\mathbf{x}_i \in \mathbb{R}^{165}$. All features are continuous and anonymized by MIT-IBM for security:

```
  ┌────────────────────────────────────────────────────────┐
  │                 165 CONTINUOUS FEATURES                │
  ├────────────────────────────┬───────────────────────────┤
  │ Local Features (1 to 93)   │ Aggregated 1-Hop (94-165) │
  │ Transacted amounts, fees,  │ Min, max, mean, std of    │
  │ in/out degree, timesteps   │ neighbor amounts & fees   │
  └────────────────────────────┴───────────────────────────┘
```

1. **Features 1 to 93 (Local Node Properties):**
   * Financial amounts sent and received (in BTC).
   * Transaction fee paid to Bitcoin miners.
   * Number of inputs (fan-in degree) and outputs (fan-out degree).
   * Timestep index and script characteristics.
2. **Features 94 to 165 (Aggregated 1-Hop Structural Features):**
   * Maximum, minimum, mean, and standard deviation of transacted amounts across all parent nodes in $\mathcal{N}(i)$.
   * Statistical dispersion of transaction fees paid by upstream and downstream neighbors.

### Zero-Leakage Feature Normalization
Because future financial distributions drift over time, z-score standardization is fitted **strictly on the training partition**:

$$\mu_{\text{train}} = \frac{1}{|\mathcal{V}_{\text{train}}|} \sum_{i \in \mathcal{V}_{\text{train}}} \mathbf{x}_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{|\mathcal{V}_{\text{train}}|} \sum_{i \in \mathcal{V}_{\text{train}}} (\mathbf{x}_i - \mu_{\text{train}})^2}$$
$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-6}}$$

---

## 6. Stratified Random Split (70 / 15 / 15)

To benchmark GAT against tabular models (Random Forest, MLP) and graph baselines (GCN), we partition the 46,564 labeled nodes using stratified random sampling (`random_state=42`):

| Partition | Share | Node Count | Licit ($y=0$) | Illicit ($y=1$) | Function in Workflow |
|:---|:---:|:---:|:---:|:---:|:---|
| **Training Set** | **70%** | **32,595** | 29,413 | 3,182 | Backpropagation loss computation & weight updates |
| **Validation Set** | **15%** | **6,985** | 6,304 | 681 | Early stopping & hyperparameter tuning |
| **Test Set** | **15%** | **6,984** | 6,302 | 682 | Unbiased final benchmark evaluation |
| **Total Labeled** | **100%** | **46,564** | 42,019 | 4,545 | Evaluated ground-truth universe |

### Class Imbalance Weighting:
Money laundering transactions make up only **9.76%** of the labeled data. To stop the network from favoring the majority licit class, the loss function applies positive class re-weighting:

$$w_{\text{pos}} = \frac{N_{\text{licit}}}{N_{\text{illicit}}} = \frac{42{,}019}{4{,}545} \approx \mathbf{9.245}$$

---

## 7. Python Loading & Split Usage

To generate and verify the dataset and split in Python:

```bash
# Generate and save split masks
python random_split.py --split_only

# Run full baseline & GAT benchmark
python random_split.py
```

Inside your code:
```python
from data_loader import load_elliptic_data

# Loads dataset with 70/15/15 split and z-score normalization
data = load_elliptic_data()

print("Train Nodes:", data.train_mask.sum().item())  # 32,595
print("Val Nodes:  ", data.val_mask.sum().item())    # 6,985
print("Test Nodes: ", data.test_mask.sum().item())   # 6,984
```
