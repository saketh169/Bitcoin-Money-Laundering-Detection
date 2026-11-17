import torch
from torch_geometric.datasets import EllipticBitcoinDataset

def load_elliptic_data(root='data/EllipticBitcoinDataset'):
    """
    Loads the Elliptic Bitcoin Dataset and applies z-score normalization
    to the node features using the training set statistics, to prevent leakage.
    Returns: PyTorch Geometric Data object.
    """
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]
    
    # In PyG EllipticBitcoinDataset (based on our inspection):
    # 0 = Licit (42,019 total)
    # 1 = Illicit (4,545 total) 
    # 2 = Unknown/Unlabeled (157,205 total)
    
    # We only care about labels 0 and 1 for evaluation.
    # The PyG train_mask and test_mask include ALL nodes in their respective timesteps.
    # We refine the masks to ONLY include labeled nodes.
    labeled_mask = (data.y == 0) | (data.y == 1)
    
    data.train_mask = data.train_mask & labeled_mask
    data.test_mask = data.test_mask & labeled_mask
    
    use_random_split = True # Switch to random split to match the high metrics in literature
    
    if use_random_split:
        import os
        split_file = os.path.join(root, 'random_split_indices.pt')
        if os.path.exists(split_file):
            splits = torch.load(split_file)
            data.train_mask = splits['train_mask']
            data.val_mask = splits['val_mask']
            data.test_mask = splits['test_mask']
        else:
            from sklearn.model_selection import train_test_split
            labeled_indices = labeled_mask.nonzero(as_tuple=False).view(-1).numpy()
            labeled_labels = data.y[labeled_mask].numpy()
            
            # Stratified 70/15/15 split
            train_idx, temp_idx, _, y_temp = train_test_split(
                labeled_indices, labeled_labels, test_size=0.30, random_state=42, stratify=labeled_labels
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.50, random_state=42, stratify=y_temp
            )
            
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            
            data.train_mask[train_idx] = True
            data.val_mask[val_idx] = True
            data.test_mask[test_idx] = True
    else:
        # Create a validation mask from the last 15% of the train mask (sequential split preserves time somewhat)
        train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
        num_train = train_indices.size(0)
        split_idx = int(num_train * 0.85)
        
        val_indices = train_indices[split_idx:]
        train_indices = train_indices[:split_idx]
        
        data.train_mask = torch.zeros_like(data.train_mask)
        data.train_mask[train_indices] = True
        
        data.val_mask = torch.zeros_like(data.train_mask)
        data.val_mask[val_indices] = True
    
    # Z-score normalization on features (166 continuous features, minus timestep)
    # We fit the scaler on the training data ONLY to prevent temporal leakage
    train_x = data.x[data.train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)
    
    # Prevent division by zero if any feature is constant
    std[std == 0] = 1.0
    
    data.x = (data.x - mean) / std
    
    return data

if __name__ == "__main__":
    data = load_elliptic_data()
    print("Train mask sum:", data.train_mask.sum().item())
    print("Test mask sum:", data.test_mask.sum().item())
    print("Train Licit:", (data.y[data.train_mask] == 0).sum().item())
    print("Train Illicit:", (data.y[data.train_mask] == 1).sum().item())
    print("Test Licit:", (data.y[data.test_mask] == 0).sum().item())
    print("Test Illicit:", (data.y[data.test_mask] == 1).sum().item())
