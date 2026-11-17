# Presentation Plan

## Slide 1: Title
**Title:** Bitcoin Money Laundering Detection Using Graph Attention Networks
**Team:** Saketh Pabbu & Nerella Venkata Sriram
*Speaker Notes:* Introduce ourselves and the topic of applying deep graph learning to financial forensics.
*Faculty Question:* Why use deep learning instead of simple rules?
*Answer:* Criminals use complex, multi-hop pathways (peeling chains) that evade simple threshold rules.

## Slide 2: Problem & Motivation
* **Bullets:**
  - Bitcoin is pseudonymous; criminals use it to launder stolen funds.
  - Evasion tactics: Peeling Chains and CoinJoins (mixing services).
  - Traditional models ignore the flow of capital.
* **Speaker Notes:* Explain how mixing services work by blending dirty money with clean money, making tabular detection very hard.

## Slide 3: Dataset & Graph Representation
* **Bullets:**
  - Elliptic Bitcoin Dataset: 203k transactions, 234k edges.
  - Nodes = Transactions (165 features). Edges = Payment flow.
  - Imbalance: Only ~2% of labeled transactions are illicit.
* **Speaker Notes:* Emphasize that we are treating the blockchain as a graph.

## Slide 4: Why GAT?
* **Bullets:**
  - Standard GCN averages all neighbors equally.
  - Small illicit signals get washed out by high-volume normal commerce.
  - Graph Attention Networks (GAT) assign dynamic weights.
* **Speaker Notes:* Give the example of 1 illicit connection hidden among 1000 licit ones. GAT can learn to pay attention only to the illicit one.
* **Faculty Question:* What exactly is the attention mechanism computing?
*Answer:* It computes an alignment score between the sender's features and the receiver's features, passing it through a LeakyReLU and softmax to get weights that sum to 1.

## Slide 5: Model Architecture & Workflow
* **Bullets:**
  - Layer 1: GATConv (4 heads) -> ELU -> Dropout (0.3)
  - Layer 2: GATConv (1 head) -> ELU -> Dropout
  - Linear classifier & Focal Loss ($\gamma=2.0$)
* **Speaker Notes:* Explain that Focal loss forces the model to learn the hard 2% minority class rather than the easy 98% majority class.

## Slide 6: Evaluation Protocol
* **Bullets:**
  - Temporal split: Train (T 1-34) | Test (T 35-49)
  - Prevents "cheating" by looking into the future.
  - Metrics: Precision, Recall, F1-Score (Accuracy is misleading).
* **Speaker Notes:* Explain why we can't use a random 80/20 split (it causes temporal leakage).

## Slide 7: Actual Results
* **Bullets:**
  - Random Forest: 0.74 F1
  - GCN Baseline: 0.39 F1
  - GAT Model: 0.44 F1
* **Speaker Notes:* Show the F1 scores. Note that GAT clearly beats GCN, proving our attention hypothesis.

## Slide 8: Error Analysis & Limitations
* **Bullets:**
  - Validation F1 was ~0.80, but dropped to 0.44 on the test set.
  - Reason: Darknet market shutdown drastically changed crime topology.
  - Random Forest was robust to this structural shift.
* **Speaker Notes:* Be honest about the drop. It's a known property of this dataset. The graph structure of crime changed, so the GNN struggled to generalize to the future.
* **Faculty Question:* If Random Forest is better, why use GNNs?
*Answer:* Random Forest is better when topology completely changes. But for static or slowly-evolving topologies, GNNs can catch complex structural patterns that tabular models miss. Future work involves Dynamic GNNs.

## Slide 9: Conclusion
* **Bullets:**
  - GATs prevent signal washout better than GCNs.
  - Future work: EvolveGCN to handle temporal graph shifts.
* **Speaker Notes:* Wrap up. Emphasize that predictions are screening signals for investigators, not proof of guilt.
