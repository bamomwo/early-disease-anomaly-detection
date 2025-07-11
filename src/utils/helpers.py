import numpy as np

# Function to compute reconstruction error for group of features.
def compute_group_errors(inputs, outputs, feature_groups):
    """
    Computes MSE reconstruction error per feature group.
    Returns a dict with group-wise errors.
    """
    group_errors = {}

    for group_name, indices in feature_groups.items():
        input_slice = inputs[:, :, indices]   # shape: (N, seq_len, num_features)
        output_slice = outputs[:, :, indices] # same

        mse = ((input_slice - output_slice) ** 2).mean()  # scalar MSE
        group_errors[group_name] = mse

    return group_errors
