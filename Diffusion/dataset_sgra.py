import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for Simulated Data
class SimulatedDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (numpy array): Array of shape (samples, timesteps, dim * 2 + 1).
                                The first `dim` entries are the ground truth values.
                                The next `dim` entries are the ground truth mask (1 for observed, 0 for masked).
                                The last entry (dim * 2 + 1) is ignored.
        """
        self.ground_truth = data[:, :, : data.shape[2] // 2]  # First dim columns
        self.gt_mask = data[:, :, data.shape[2] // 2 : -1]  # Next dim columns
        self.timepoints = np.arange(data.shape[1])  # Create a range for timesteps
        self.times = np.arange(data.shape[1])

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        observed_data = self.ground_truth[idx]
        observed_mask = np.ones_like(observed_data)  # All ones, since all data is observed
        gt_mask = self.gt_mask[idx]
        timepoints = self.timepoints
        times = self.times

        return {
            "observed_data": torch.tensor(observed_data, dtype=torch.float32),
            "observed_mask": torch.tensor(observed_mask, dtype=torch.float32),
            "gt_mask": torch.tensor(gt_mask, dtype=torch.float32),
            "timepoints": torch.tensor(timepoints, dtype=torch.float32),
            "times": torch.tensor(times, dtype=torch.float32)
        }
    

def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    # Load the .npz file
    file_path = "data/sgra.npz"  # Update with your file path
    data = np.load(file_path)

    # Extract train, validation, and test datasets
    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    # Create Dataset instances
    train_dataset = SimulatedDataset(train_data)
    val_dataset = SimulatedDataset(val_data)
    test_dataset = SimulatedDataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# # Example: Accessing a batch
# for batch in train_loader:
#     print(batch["observed_data"].shape)
#     print(batch["observed_mask"].shape)
#     print(batch["gt_mask"].shape)
#     print(batch["timepoints"].shape)
#     break
