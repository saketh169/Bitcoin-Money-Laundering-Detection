# Report Draft: Bitcoin Money Laundering Detection Using Graph Attention Networks

**Team:** Saketh Pabbu (S20230010169), Nerella Venkata Sriram (S20230010164)
**Course:** Advanced Deep Learning

## 1. Abstract
Money laundering detection is a critical challenge in financial forensics. Transactions on the blockchain naturally form a graph, where individual transactions correspond to nodes and the flow of cryptocurrency represents directed edges. This project investigates the application of Graph Attention Networks (GAT) for detecting illicit transactions in the Bitcoin blockchain. A GAT-based model is implemented using PyTorch Geometric and evaluated on the benchmark Elliptic dataset. To prevent temporal data leakage, data is partitioned by time (Timesteps 1-34 for training, 35-49 for testing). Because only ~2% of transactions are criminal, the model is trained using Focal Loss. Its performance is evaluated against Random Forest, MLP, and standard GCN.

## 2. Problem Statement
In the Bitcoin network, criminal syndicates exploit pseudonymous addresses to wash stolen cryptocurrency. They rely on evasion strategies like peeling chains and mixing services (CoinJoins). Traditional machine learning models inspect transactions in isolation. Standard Graph Convolutional Networks (GCN) treat every connection equally, allowing small illicit signals to get washed out by normal commerce. We implemented a GAT model to trace multi-hop payment trails and assign attention to suspicious flows.

## 3. Dataset and Preprocessing
The Elliptic dataset consists of 49 distinct time-steps.
* **Total Transactions:** 203,769 nodes
* **Total Edges:** 234,355 payment links
* **Features:** 165 continuous numerical features per node.
* **Labels:** Licit (Class 0), Illicit (Class 1), Unknown (Class 2).
* **Split:** Training on timesteps 1-34. Testing on timesteps 35-49. The last 15% of the training split was held out as a validation set for model selection.
* **Preprocessing:** Unknown nodes are masked out of the loss calculation. The 165 continuous features were standardized using z-score normalization fit purely on the training data.

## 4. Methodology
We implemented a 2-layer Graph Attention Network (GAT):
* **Layer 1:** 4 attention heads, outputting to a hidden dimension of 64.
* **Layer 2:** 1 attention head.
* **Activations & Regularization:** ELU activation and Dropout (p=0.3) between layers.
* **Loss:** Focal Loss ($\gamma = 2.0$) to counter severe class imbalance.
* **Optimizer:** Adam (learning rate = 0.001).

## 5. Experimental Results
Metrics were calculated on the held-out future test set (timesteps 35-49) to simulate realistic forensic deployment.

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.9938 | 0.5946 | **0.7441** | 0.9226 |
| MLP | 0.4403 | 0.2724 | 0.3366 | 0.8771 |
| GCN | 0.3610 | 0.4414 | 0.3972 | 0.8312 |
| GAT | 0.4075 | 0.4940 | 0.4466 | 0.8678 |

## 6. Discussion and Error Analysis
Our experiments yielded two primary findings:
1. **GAT outperforms GCN:** As hypothesized, GAT (0.44 F1) outperformed standard GCN (0.39 F1). The attention mechanism successfully prevents small illicit signals from being washed out by high-volume benign neighborhoods.
2. **Temporal Generalization Drop:** While the GAT achieved a high F1-score (~0.80) on the validation set from historical timesteps, its performance dropped on the future test set. Interestingly, the tabular Random Forest model was far more resilient to this temporal shift (0.74 F1). This is a well-documented phenomenon in the Elliptic dataset caused by a darknet market shutdown around timestep 43. Topological patterns of crime changed drastically, causing GNNs to struggle with future generalization compared to local feature heuristics.

## 7. Conclusion
This project successfully implemented and evaluated a Graph Attention Network for anti-money laundering. While GATs provide a strong improvement over traditional graph convolutions by dynamically weighting suspicious paths, they remain highly sensitive to evolving criminal behaviors over time. Future work should investigate dynamic or evolving GNNs (like EvolveGCN) to adapt to changing laundering topologies.

## References
[1] M. Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics," arXiv:1908.02591, 2019.
[2] P. Veličković et al., "Graph Attention Networks," ICLR, 2018.
