import os
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

def main():
    os.makedirs('results', exist_ok=True)
    
    print("Downloading/Loading Elliptic Bitcoin Dataset...")
    dataset = EllipticBitcoinDataset(root='data/EllipticBitcoinDataset')
    
    data = dataset[0]  # The dataset is a single large temporal graph
    
    with open('results/data_inspection.txt', 'w') as f:
        def log(msg):
            print(msg)
            f.write(msg + '\n')
            
        log(f"Dataset: {dataset}")
        log(f"Number of graphs: {len(dataset)}")
        log(f"\n--- Graph Statistics ---")
        log(f"Nodes (Transactions): {data.num_nodes}")
        log(f"Edges: {data.num_edges}")
        log(f"Node Features: {data.num_node_features}")
        log(f"Classes: {dataset.num_classes}")
        
        log(f"\n--- Tensors ---")
        log(f"x shape: {data.x.shape} (Node features)")
        log(f"edge_index shape: {data.edge_index.shape} (Edges)")
        log(f"y shape: {data.y.shape} (Labels)")
        if hasattr(data, 'train_mask'):
            log(f"train_mask shape: {data.train_mask.shape}")
        
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        log(f"\n--- Label Distribution ---")
        for label, count in zip(unique_labels, counts):
            log(f"Label {label.item()}: {count.item()}")
            
        log("\nNote: In PyTorch Geometric's EllipticBitcoinDataset, labels are typically mapped as:")
        log("Label 0: Illicit")
        log("Label 1: Licit")
        log("Label 2: Unknown (unlabeled)")
        
        # Check time steps if available
        if hasattr(data, 'time_step'):
            log(f"\n--- Time Steps ---")
            log(f"time_step shape: {data.time_step.shape}")
            min_time = data.time_step.min().item()
            max_time = data.time_step.max().item()
            log(f"Time steps range from {min_time} to {max_time}")

if __name__ == "__main__":
    main()
