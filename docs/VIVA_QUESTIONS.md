# Viva Preparation: 20 Questions

1. **Why this problem and dataset?**
   *Answer:* Money laundering via cryptocurrency is a major issue. The Elliptic dataset is the standard benchmark for this, allowing us to model the blockchain natively as a graph rather than forcing it into a tabular format.

2. **What is a graph in this project?**
   *Answer:* Nodes are Bitcoin transactions. Edges represent the flow of Bitcoin from one transaction to another. Features are transaction metadata (like amounts and fees).

3. **Why use GAT instead of only a tabular model?**
   *Answer:* Tabular models ignore how money flows. Criminals use complex paths (like peeling chains) to hide money. GAT looks at these multi-hop paths to identify structural patterns of crime.

4. **What is the difference between GCN and GAT?**
   *Answer:* GCN averages the features of all neighbors equally. GAT calculates an attention score for each neighbor, allowing it to give more weight to suspicious neighbors and ignore benign ones.

5. **How did GAT perform compared to GCN in your actual runs?**
   *Answer:* GAT outperformed GCN (0.44 vs 0.39 F1-score), proving that the attention mechanism successfully stops illicit signals from being washed out by large volumes of normal commerce.

6. **What are attention heads?**
   *Answer:* Multi-head attention means the model computes the attention weights multiple times independently (we used 4 heads). This allows the model to focus on different aspects of a neighbor simultaneously.

7. **How did you handle the severe class imbalance (2% illicit)?**
   *Answer:* We used Focal Loss with $\gamma = 2.0$. It dynamically scales the loss based on prediction confidence, forcing the optimizer to focus heavily on the hard-to-predict illicit minority class.

8. **Why is Accuracy a bad metric for this project?**
   *Answer:* Because 98% of the data is Licit. A dummy model that just predicts "Licit" every time would achieve 98% accuracy but catch exactly zero criminals.

9. **What do Precision and Recall mean here?**
   *Answer:* Precision: Out of all transactions we flagged as illicit, how many actually were? Recall: Out of all truly illicit transactions, how many did we manage to catch?

10. **Explain your train/test split.**
    *Answer:* We trained strictly on timesteps 1 to 34 and tested on 35 to 49. We used a sequential chunk of the train split as validation.

11. **Why not use an 80/20 random split?**
    *Answer:* Random splitting causes temporal leakage. The model would learn from future transactions to predict past ones, which is impossible in the real world.

12. **Why did the model's performance drop on the test set?**
    *Answer:* Around timestep 43, a major darknet market was shut down. Criminals changed their money laundering topologies drastically. The GNN memorized old topologies and struggled to generalize to the new ones.

13. **Why did Random Forest beat the GNNs on the test set?**
    *Answer:* Random Forest only relies on local features (like transaction amounts). These features remained relatively stable even when the overall network topology (which the GNN relies on) changed entirely.

14. **How are unknown nodes handled?**
    *Answer:* They are included in the graph so the model can pass messages through them, but they are masked out of the loss calculation so we don't penalize the model for predictions on them.

15. **What is ROC-AUC?**
    *Answer:* It measures the model's ability to rank illicit transactions higher than licit ones across all possible classification thresholds.

16. **Did you use GPU or CPU for your runs?**
    *Answer:* We ran this on CPU. PyTorch Geometric is optimized enough to handle graphs of 200k nodes on a CPU, though it takes several minutes to train.

17. **What activation function did you use?**
    *Answer:* ELU (Exponential Linear Unit).

18. **Why apply Z-score normalization only on the training set?**
    *Answer:* If we calculated the mean and standard deviation using the test set, information from the future test set would "leak" into the training features.

19. **What is a False Positive in this context, and why is it dangerous?**
    *Answer:* A False Positive is flagging a legitimate user as a criminal. It is dangerous because it could lead to frozen funds, account suspensions, and legal trouble for innocent people.

20. **Does your model prove someone is a criminal?**
    *Answer:* No. A high probability score is merely a screening signal. It flags suspicious clusters for human compliance officers to investigate manually.
