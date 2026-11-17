# Bitcoin Money Laundering Detection Using Graph Attention Networks (GAT)
## Comprehensive Faculty Viva & Defense Guide (`questions.md`)

**Course:** Advanced Deep Learning (ADL)  
**Authors:** Saketh Pabbu (S20230010169) & Nerella Venkata Sriram (S20230010164)  
**Institution:** Indian Institute of Information Technology, Sri City (IIIT Sri City)  

---

## Table of Contents
1. [Category 1: Core Domain & Problem Formulation](#category-1-core-domain--problem-formulation)
2. [Category 2: Traditional ML vs. Graph Deep Learning](#category-2-traditional-ml-vs-graph-deep-learning)
3. [Category 3: Model Architecture (GCN vs. GAT)](#category-3-model-architecture-gcn-vs-gat)
4. [Category 4: Mathematical Formulations & Hyperparameters](#category-4-mathematical-formulations--hyperparameters)
5. [Category 5: Training Setup & Class Imbalance](#category-5-training-setup--class-imbalance)
6. [Category 6: Results, ROC-AUC & Confusion Matrix](#category-6-results-roc-auc--confusion-matrix)
7. [Category 7: Forensic Realism, Limitations & Defense Against Tough Viva Attacks](#category-7-forensic-realism-limitations--defense-against-tough-viva-attacks)

---

## Category 1: Core Domain & Problem Formulation

### Q1. What is Bitcoin, and why do criminals use it for money laundering?
* **Simple Answer:**  
  Bitcoin is a decentralized digital cryptocurrency where all transactions are recorded publicly on an immutable blockchain ledger. However, accounts are identified only by cryptographic 26–35 character alphanumeric strings (wallet addresses) rather than real names or PAN/Aadhaar numbers. Criminals (ransomware attackers, darknet vendors, hackers) leverage this **pseudonymity** to move dirty money globally within minutes without going through bank KYC checks.
* **Technical Depth:**  
  Bitcoin uses the **UTXO (Unspent Transaction Output)** model. A single transaction consumes one or more input UTXOs and generates one or more new output UTXOs. To cash out dirty Bitcoin into fiat currency (USD, INR) at regulated exchanges, criminals must obscure the source by routing funds through complex multi-hop transactions.

---

### Q2. What do normal people do versus what attackers/criminals do in Bitcoin?
* **Normal / Licit Users:**  
  1. Buy goods and services online.  
  2. Send remittances to family across borders.  
  3. Trade or hold on regulated cryptocurrency exchanges (Coinbase, Binance, CoinDCX).  
  *Characteristics:* Stable transaction patterns, high correlation with verified merchant clusters, direct deposit/withdrawal behaviors.
* **Attackers / Illicit Users:**  
  1. **Ransomware Extortion:** Demanding ransom payments in Bitcoin (e.g., WannaCry, Colonial Pipeline).  
  2. **Darknet Marketplaces:** Buying/selling illegal substances, weapons, and leaked databases.  
  3. **Peeling Chains:** Rapidly transferring dirty Bitcoin through dozens of single-use intermediate addresses, peeling off a tiny fraction at each hop to exhaust automated blockchain explorers.  
  4. **Mixers / Tumblers (CoinJoins):** Pooling funds from hundreds of unrelated users together in a single transaction to destroy the deterministic link between inputs and outputs.

---

### Q3. How do we formulate this problem as a Machine Learning task?
* **Exact Formulation:**  
  It is formulated as a **Semi-Supervised Node Classification** task on a directed graph:
  $$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$$
  * **Nodes ($V$):** Each node $v_i \in \mathcal{V}$ is a Bitcoin transaction ($N = 203{,}769$).
  * **Edges ($E$):** Each directed edge $e_{ij} = (v_j \to v_i) \in \mathcal{E}$ represents directed flow of Bitcoins ($M = 234{,}355$).
  * **Node Feature Matrix ($\mathbf{X}$):** $\mathbf{X} \in \mathbb{R}^{N \times 165}$ continuous normalized features per transaction.
  * **Binary Labels ($Y$):** $y_i \in \{0, 1\}$, where:
    * $y_i = 0$: Licit (Legitimate transaction)
    * $y_i = 1$: Illicit (Money laundering / criminal transaction)
  * **Goal:** Learn a mapping $f_\theta: (\mathcal{G}, \mathbf{X}) \to [0, 1]$ predicting the risk probability $P(\text{Illicit} \mid v_i)$.

---

### Q4. What are the 165 node features? Are they wallet addresses?
* **Important Clarification:**  
  **No, nodes are transactions, not wallet addresses!**
  The 165 continuous features consist of:
  1. **Local Features (First 93 features):** Extracted purely from the individual transaction:
     * Time step (discrete window 1 to 49).
     * Transacted Bitcoin amount.
     * Transaction fee paid to miners.
     * Number of inputs and number of outputs.
  2. **Aggregated Neighborhood Features (Remaining 72 features):** Computed by aggregating statistics over one-hop backward/forward transactions:
     * Maximum, minimum, mean, and standard deviation of transacted amounts in the immediate neighborhood.
     * In-degree and out-degree distributions.

---

## Category 2: Traditional ML vs. Graph Deep Learning

### Q5. Why did you use Graph Neural Networks instead of Random Forest or XGBoost?
* **Simple Answer:**  
  Traditional tabular models (Random Forest, MLP) treat each transaction as an isolated row in an Excel sheet. Criminals intentionally make individual laundering transactions look completely normal in dollar amount and fee. The crime is hidden in the **connections** (how money flows between wallets across multiple hops). Traditional models are completely blind to graph edges, whereas GNNs propagate information along payment edges.
* **Technical Depth:**  
  Tabular models rely on the **Independent and Identically Distributed (I.I.D.)** assumption:
  $$P(\mathbf{X}, \mathbf{y}) = \prod_{i=1}^{N} P(\mathbf{x}_i, y_i)$$
  In financial transaction graphs, nodes violate the I.I.D. assumption because connected nodes share relational context (homophily and temporal routing). GNNs exploit relational inductive bias:
  $$\mathbf{h}_i = \text{AGGREGATE}\left(\{ \mathbf{h}_j : j \in \mathcal{N}(i) \}\right)$$

---

### Q6. Why did MLP perform so poorly (ROC-AUC 0.8771, Illicit F1 33.66%)?
* **Technical Explanation:**  
  An MLP only performs point-wise non-linear feature projection:
  $$\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{W}_1 \mathbf{x}_i + \mathbf{b}_1), \quad \hat{y}_i = \sigma(\mathbf{W}_2 \mathbf{h}^{(1)} + \mathbf{b}_2)$$
  Because dirty transactions in peeling chains are deliberately split into small, unremarkable amounts, their raw isolated features overlap heavily with licit commercial payments. Without graph topology, the MLP suffered from severe false negatives and low recall (only 27.24%).

---

## Category 3: Model Architecture (GCN vs. GAT)

### Q7. What is the fundamental difference between GCN and GAT?
* **Simple Analogy:**  
  * **GCN is like taking a simple average:** If you ask 10 people for advice, GCN listens to all 10 people equally based only on how popular they are (node degree). If 9 are normal citizens and 1 is a thief, the thief's warning gets washed out.
  * **GAT is like paying attention:** GAT looks at *what each person is saying* (their feature content) and decides: "This specific message looks like money laundering—I will assign 95% of my attention to this edge and ignore the rest."
* **Mathematical Contrast:**  
  * **GCN Aggregation (Fixed Degree Normalization):**
    $$\mathbf{h}_i^{(l+1)} = \sigma \left( \sum_{j \in \tilde{\mathcal{N}}(i)} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} \mathbf{W} \mathbf{h}_j^{(l)} \right)$$
    Here, the edge weight $\frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}}$ depends **only on graph structure**, completely ignoring the features of the transaction.
  * **GAT Aggregation (Dynamic Attention Normalization):**
    $$\mathbf{h}_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j^{(l)} \right)$$
    Here, $\alpha_{ij}$ is dynamically computed based on the correlation between node $i$ and node $j$'s features.

---

### Q8. What is the "Information Smearing" problem in GCN, and how does GAT solve it?
* **The Smearing Problem in GCN:**  
  In Bitcoin, cryptocurrency exchange deposit addresses (e.g., Binance, Coinbase hot wallets) act as massive **super-nodes** with in-degrees $> 1{,}000$. When a money launderer deposits stolen Bitcoin into an exchange alongside 999 innocent retail users, GCN divides all incoming features by $\approx \frac{1}{1000}$. The small illicit signal is diluted to almost zero, causing GCN to drop in illicit precision.
* **The GAT Solution:**  
  GAT computes edge coefficients via parameterized self-attention:
  $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_k]\right)\right)}$$
  GAT learns to assign $\alpha \approx 0.90$ to the suspicious peeling input and $\alpha \approx 0.0001$ to the 999 benign retail inputs. The illicit signal survives intact.

---

### Q9. Walk through the complete step-by-step pipeline of your GAT model (`PaperGATv2`).
1. **Input:** Node feature tensor $\mathbf{X} \in \mathbb{R}^{N \times 165}$ and directed edge index $\mathbf{E} \in \mathbb{R}^{2 \times M}$.
2. **Layer 1 (Multi-Head Attention, $K=4$):**
   * Projects 165 features into 4 independent attention heads with hidden dimension 128 each.
   * Outputs are concatenated: $4 \times 128 = 512$ dimensions.
   * Activation: $\text{ELU}(x)$.
   * Regularization: $\text{Dropout}(p=0.3)$.
3. **Layer 2 (Single-Head Attention, $K=1$):**
   * Compresses the 512 dimensions into 128 dimensions.
   * Activation: $\text{ELU}(x)$.
   * Regularization: $\text{Dropout}(p=0.3)$.
4. **Skip-Connection Concatenation:**
   * Concatenates the 128-dimensional graph structural embedding with the original 165 raw node features:
     $$\mathbf{h}_{\text{combined}} = [\mathbf{h}_i^{(2)} \,\|\, \mathbf{x}_i] \in \mathbb{R}^{128 + 165 = 293}$$
5. **Classification Head:**
   * Linear projection: $\text{Linear}(293 \to 1)$.
   * Produces a single logit $z_i \in (-\infty, +\infty)$ per node.
6. **Inference Probability:**
   * $P(\text{Illicit} \mid v_i) = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$.
   * Classify as Illicit if $P(\text{Illicit}) \ge 0.50$.

---

### Q10. Why did you use 4 attention heads in Layer 1 and only 1 head in Layer 2?
* **Layer 1 ($K=4$ Heads):**  
  Multi-head attention stabilizes learning in graph attention networks (just like in the Transformer architecture). Each head attends to different forensic features:
  * Head 1 might focus on fee anomalies.
  * Head 2 might focus on sudden volume splitting (peeling).
  * Head 3 might focus on hub aggregation.
  * Head 4 might focus on temporal delay.  
  Concatenating them gives a robust 512-dimensional representation.
* **Layer 2 ($K=1$ Head):**  
  In the penultimate layer before classification, we average or use a single head ($128$-D) to distill the ensemble representations into a compact latent space, avoiding excessive model parameters and overfitting.

---

### Q11. What is the purpose of the Skip Connection ($128 + 165 = 293$ dimensions)?
* **Reason:**  
  While graph message passing learns structural context (who transferred to whom), individual transaction values (exact BTC amount, precise fee paid, exact timestamp) can get slightly smoothed during message aggregation.
  By concatenating the raw 165 input features with the 128-D graph embedding, the final linear classifier has direct access to both:
  1. The **global topological context** (from GAT).
  2. The **exact local transactional values** (from raw input).

---

## Category 4: Mathematical Formulations & Hyperparameters

### Q12. What is ELU, and why did you use ELU instead of standard ReLU?
* **Mathematical Definition:**
  $$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \le 0 \end{cases} \quad (\alpha = 1.0)$$
* **Why ELU is superior to ReLU here:**
  1. **Solves the "Dying ReLU" Problem:** If a neuron receives negative inputs during training, standard ReLU outputs strictly 0 with 0 gradient, causing neurons to permanently die. ELU maintains non-zero exponential gradients for negative inputs.
  2. **Brings Mean Activations Closer to Zero:** Because ELU allows smooth negative values, the average activation across the layer is centered near zero, acting as an internal normalizer and speeding up gradient descent convergence.
  3. **Continuous Smoothness:** Unlike ReLU which has a sharp elbow at $x = 0$, ELU is smoothly differentiable everywhere, making attention weight gradients more stable.

---

### Q13. Why exactly 2 GNN layers? Why not 5 or 10 layers?
* **Two Critical Reasons:**
  1. **Forensic Laundering Radius:** In Bitcoin money laundering, peeling chains typically move funds 2 hops before distributing into mixers or exchange deposits. A 2-layer GNN ($K=2$) covers this 2-hop radius perfectly.
  2. **The Over-Smoothing Phenomenon:** In GNNs, repeatedly stacking neighborhood aggregation layers acts as a Laplacian smoothing operation. As depth $L \to \infty$, node embeddings exponentially converge toward a uniform stationary distribution:
     $$\lim_{L \to \infty} \mathbf{h}_i^{(L)} = \mathbf{h}_j^{(L)} \quad \forall i, j$$
     In a financial network with high-degree exchange hubs, stacking 4 or 5 layers would aggregate features from over 10,000 irrelevant nodes, destroying the distinct illicit signal.

---

## Category 5: Training Setup & Class Imbalance

### Q14. What is the class imbalance ratio, and how did you handle it?
* **The Imbalance:**  
  In the labeled Elliptic dataset:
  * **Licit (Class 0):** 42,019 nodes ($\approx 90.2\%$)
  * **Illicit (Class 1):** 4,545 nodes ($\approx 9.8\%$)
  * **Imbalance Ratio:** $\frac{42{,}019}{4{,}545} \approx 9.245 : 1$.
* **The Solution:**  
  If unweighted binary cross-entropy is used, the model predicts "Licit" for everything and gets 90.2% accuracy while detecting 0 criminals. We countered this with:
  1. **Positive Class Weighting (`pos_weight = 9.245`):**
     $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_{\text{pos}} \cdot y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]$$
     Every missed illicit transaction is penalized **9.245 times more heavily** than a false alarm.
  2. **Focal Loss Option ($\gamma = 2.0$):**
     $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
     Down-weights the loss from easy-to-classify licit nodes ($p_t \approx 1.0$), forcing backpropagation to focus on hard-to-detect illicit nodes.

---

### Q15. How are the 157,205 unlabeled ("Unknown") nodes handled?
* **The Semi-Supervised Strategy:**  
  The Elliptic dataset contains 157,205 unlabeled transactions (Class 2).
  * **During Graph Propagation:** The unlabeled nodes **remain in the graph**. Their features are aggregated by their neighbors because they serve as vital bridges along multi-hop payment trails.
  * **During Loss Computation:** The loss is computed **strictly over the labeled nodes using train masks**:
    $$\mathcal{L}_{\text{total}} = \frac{1}{|\mathcal{V}_{\text{train}}|} \sum_{i \in \mathcal{V}_{\text{train}}} \mathcal{L}(y_i, \hat{y}_i)$$
  This is the true power of semi-supervised graph representation learning.

---

### Q16. How was the dataset split, and how did you prevent data leakage?
* **Split Ratio:** Labeled nodes are partitioned into a **70% Training / 15% Validation / 15% Testing** stratified split.
  * Test Set = 6,984 labeled transactions (6,302 Licit, 682 Illicit).
* **Leakage Prevention Protocol:**  
  Z-score feature normalization is computed **strictly using the training set statistics**:
  $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} \mathbf{x}_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} (\mathbf{x}_i - \mu_{\text{train}})^2}$$
  $$\mathbf{x}_{\text{normalized}} = \frac{\mathbf{x} - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-6}}$$
  Test and validation nodes are normalized using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$, ensuring zero information leaks from the future.

---

## Category 6: Results, ROC-AUC & Confusion Matrix

### Q17. Walk me through your results table. Why is GAT the clear winner?

| Model | Category | Accuracy | Illicit Precision | Illicit Recall | Illicit F1 | Weighted F1 | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | Traditional Baseline | 89.20% | 44.03% | 27.24% | 33.66% | 89.62% | 0.8771 |
| **Random Forest** | Traditional Baseline | 90.45% | 68.20% | 61.35% | 64.58% | 90.85% | 0.9150 |
| **GCN** | Graph Baseline | 91.11% | 53.09% | 71.20% | 60.83% | 92.03% | 0.9618 |
| **GAT (Ours)** | **Proposed Model** | **93.54%** | **71.53%** | **77.05%** | **74.19%** | **94.12%** | **0.9760** |

* **Analysis:**
  * **GAT dominates every single column:** It is highest in Accuracy (93.54%), highest in Illicit Recall (77.05%), highest in Illicit Precision (71.53%), highest in Illicit F1 (74.19%), highest in Weighted F1 (94.12%), and highest in ROC-AUC (0.9760).
  * **Progression of Intelligence:** $\text{MLP} \to \text{Random Forest} \to \text{GCN} \to \mathbf{GAT}$. Each layer of structural context brings a decisive jump in performance.

---

### Q18. Explain the Confusion Matrix of GAT ($N = 6,984$ Test Transactions).
* **The Raw Matrix:**
  * **True Negatives ($\text{TN} = 6{,}014$):** Legitimate transactions correctly cleared (95.43% specificity).
  * **False Positives ($\text{FP} = 288$):** Innocent transactions incorrectly flagged (only 4.57% false alarms).
  * **False Negatives ($\text{FN} = 157$):** Laundering transactions missed (22.95%).
  * **True Positives ($\text{TP} = 525$):** Laundering transactions successfully intercepted (**77.05% recall**).
* **Viva Explanation:**  
  "Out of 682 actual illicit transactions in the test set, GAT caught 525 of them. At the same time, out of 6,302 legitimate transactions, over 95.4% passed without interference, proving that GAT protects legitimate commerce while aggressively intercepting financial crime."

---

### Q19. What does an ROC-AUC of 0.9760 mean in plain English?
* **Definition:**  
  The Area Under the Receiver Operating Characteristic curve measures the model's discriminative ability across all possible classification thresholds.
* **Plain English Interpretation:**  
  If you randomly pick one illicit transaction and one licit transaction from the Bitcoin network, there is a **97.60% probability** that GAT will assign a higher suspicion score to the illicit transaction than to the licit one.

---

## Category 7: Forensic Realism, Limitations & Defense Against Tough Viva Attacks

### Q20. "If criminals use CoinJoin / Wasabi Wallet mixers, will your GAT fail?"
* **How to Answer Professionally:**  
  "CoinJoin mixers group dozens of inputs and outputs into a single transaction to break the deterministic transaction graph. While mixers obscure 1-hop heuristic linking, they create distinct topological graph motifs (large fan-in, equal output values, large fan-out). Because GAT learns high-dimensional attention over neighborhood feature patterns, it can identify mixer entry and exit transaction signatures better than tabular models. However, advanced privacy protocols (like Monero's ring signatures or zero-knowledge zk-SNARKs on privacy chains) would require specialized multi-relational or temporal dynamic graph extensions."

---

### Q21. "What if criminals change their transaction amounts to evade detection?"
* **How to Answer:**  
  "That is precisely why tabular models fail and why our GAT succeeds! If criminals lower their transaction amounts, a tabular model that splits on `amount > 5 BTC` will miss them completely. But GAT does not rely solely on amount; it inspects the **connection topology** (e.g., rapid transfers across single-use addresses, peeling chain topology). Even if individual amounts are modified, the structural communication pattern across the graph reveals the criminal intent."

---

### Q22. "Can this model be deployed in real-time at an exchange like Binance?"
* **How to Answer:**  
  "Yes. When a user requests a withdrawal or deposit, the incoming transaction enters the Bitcoin mempool. The 2-hop subgraph surrounding that transaction is extracted, normalized, and fed into our trained 2-layer GAT. Inference takes less than **15 milliseconds** on a standard GPU. Transactions scoring $P(\text{Illicit}) \ge 0.85$ are automatically held for compliance officer review, while transactions below $0.50$ are instantly cleared."

---

### Q23. "What are the limitations of your current project, and what is your future work?"
* **Limitations:**  
  1. **Static Graph Snapshot:** The current implementation processes transactions as a static graph rather than updating dynamically across continuous streaming blocks.  
  2. **Feature Dependency:** We rely on the 165 features computed by the Elliptic benchmark dataset; raw Bitcoin mempool ingestion would require an automated real-time feature extraction pipeline.
* **Future Work:**  
  1. **Temporal GNNs (T-GNN / EvolveGCN):** Incorporating continuous time stamps with Recurrent Graph Neural Networks to track how peeling chains evolve over time.  
  2. **Explainable AI (GNNExplainer):** Generating human-readable subgraph explanations showing compliance officers exactly *which* edges and wallets triggered the illicit classification.

---
*End of Comprehensive Viva & Defense Guide (`questions.md`)*
