import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysiologicalDataset(Dataset):
    """
    Custom Dataset class for physiological data with temporal sequences.
    
    This dataset handles:
    - Loading normalized filled data, and masks
    - Creating temporal sequences from windowed features
    - Handling missing data through masking
    - Supporting both single and multiple participant training
    """
    
    def __init__(
        self,
        data_path: str,
        participants: List[str],
        sequence_length: int = 24,
        overlap: float = 0.2,
        data_type: str = 'train',
        normalize: bool = True,
        return_participant_id: bool = True,
        selected_features_path: Optional[str] = None,
        filter_stress: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Base path to data directory
            participants: List of participant IDs (e.g., ['001', '002'])
            sequence_length: Number of windows per sequence
            overlap: Overlap between consecutive sequences (0.0 to 1.0)
            data_type: 'train' or 'test'
            normalize: Whether to apply normalization
            return_participant_id: Whether to return participant ID with each sample
            filter_stress: If True, only load stress-free (level 0) data.
        """
        self.data_path = Path(data_path)
        self.participants = participants
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.data_type = data_type
        self.normalize = normalize
        self.return_participant_id = return_participant_id
        self.filter_stress = filter_stress
        # ── load your JSON list of "features we actually want" ──
        if selected_features_path:
            with open(selected_features_path, 'r') as f:
                self.selected_features = json.load(f)['features']
        else:
            self.selected_features = None
        
        # Calculate step size for sequence creation
        self.step_size = max(1, int(sequence_length * (1 - overlap)))
        
        # Initialize data containers
        self.sequences = []
        self.masks = []
        self.participant_ids = []
        
        # Load and process data
        self._load_data()
        
        logger.info(f"Dataset initialized with {len(self.sequences)} sequences")
        logger.info(f"Sequence length: {self.sequence_length}, Step size: {self.step_size}")
    
    def _load_data(self):
        """Load and process data for all participants."""
        successful_loads = 0
        for participant in self.participants:
            if self._load_participant_data(participant):
                successful_loads += 1
        
        if successful_loads == len(self.participants):
            logger.info(f"Successfully loaded {self.data_type.upper()} data for all {len(self.participants)} participants.")
        else:
            logger.warning(
                f"Loaded {self.data_type.upper()} data for {successful_loads} out of {len(self.participants)} participants. "
                f"{len(self.participants) - successful_loads} failed."
            )
        logger.info("")
    
    def _load_participant_data(self, participant: str):
        """Load data for a single participant from the new normalized structure."""
        try:
            logger.info(f"─" * 15 + f" Processing participant: {participant} ({self.data_type.upper()}) " + f"─" * 15)
            # Construct file paths for the new structure
            filled_path = self.data_path / self.data_type / 'filled' / f'{participant}_{self.data_type}_filled.csv'
            mask_path = self.data_path / self.data_type / 'mask' / f'{participant}_{self.data_type}_mask.csv'

            logger.debug(f"Loading filled data from: {filled_path}")
            logger.debug(f"Loading mask from: {mask_path}")

            # Check if files exist
            if not filled_path.exists():
                logger.error(f"Filled data file not found: {filled_path}")
                return False
            if not mask_path.exists():
                logger.error(f"Mask data file not found: {mask_path}")
                return False

            # Load filled data (NaNs replaced with zeros)
            filled_data_ = pd.read_csv(filled_path)

            # Load mask data (now as CSV for each split)
            mask_data_ = pd.read_csv(mask_path)

            # Filter for stress-free samples if required
            if self.filter_stress:
                stress_free_indices = filled_data_[filled_data_['stress_level'] == 0].index
                filled_data_ = filled_data_.loc[stress_free_indices].reset_index(drop=True)
                mask_data_ = mask_data_.loc[stress_free_indices].reset_index(drop=True)

            # Drop non-feature columns
            to_drop = [c for c in ('session','timestamp','stress_level') if c in filled_data_.columns]
            filled_data_ = filled_data_.drop(columns=to_drop)

            # ── subset to only your selected features ──
            if self.selected_features:
                missing = set(self.selected_features) - set(filled_data_.columns)
                if missing:
                    logger.error(f"Selected features not found for {participant}: {missing}")
                filled_data_ = filled_data_[self.selected_features]
            filled_data = filled_data_

            # Drop the same non-feature columns from the already-filtered mask
            mask_data_ = mask_data_.drop(columns=to_drop, errors='ignore')

            # ── subset mask to the same selected features ──
            if self.selected_features:
                mask_data_ = mask_data_[self.selected_features]
            
            mask_data = mask_data_.values.astype(np.bool_)

            # Ensure mask and data have same shape
            if filled_data.shape != mask_data.shape:
                logger.warning(
                    f"Shape mismatch for participant {participant}: "
                    f"data {filled_data.shape}, mask {mask_data.shape}"
                )
                return

            # Convert to numpy arrays
            filled_array = filled_data.values.astype(np.float32)
            mask_array = mask_data

            # Create sequences from windows
            sequences, masks = self._create_sequences(filled_array, mask_array)

            # Store sequences
            self.sequences.extend(sequences)
            self.masks.extend(masks)
            self.participant_ids.extend([participant] * len(sequences))

            logger.info(f"Loaded {len(sequences)} sequences for participant {participant}")

            return True

        except Exception as e:
            logger.error(f"Error loading data for participant {participant}: {str(e)}")
            return False
    
    def _create_sequences(self, data: np.ndarray, mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create overlapping sequences from windowed data.
        
        Args:
            data: Array of shape (n_windows, n_features)
            mask: Boolean mask of shape (n_windows, n_features)
            
        Returns:
            Tuple of (sequences, masks) lists
        """
        sequences = []
        masks = []
        
        n_windows = data.shape[0]
        
        # Create sequences with sliding window approach
        for start_idx in range(0, n_windows - self.sequence_length + 1, self.step_size):
            end_idx = start_idx + self.sequence_length
            
            # Extract sequence and corresponding mask
            sequence = data[start_idx:end_idx]  # Shape: (sequence_length, n_features)
            sequence_mask = mask[start_idx:end_idx]  # Shape: (sequence_length, n_features)
            
            sequences.append(sequence)
            masks.append(sequence_mask)
        
        return sequences, masks
    
    def __len__(self) -> int:
        """Return the total number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sequence and its associated data.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary containing:
            - 'data': Tensor of shape (sequence_length, n_features)
            - 'mask': Boolean tensor of shape (sequence_length, n_features)
            - 'participant_id': String (if return_participant_id=True)
        """
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
        
        # Get sequence and mask
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        
        # Create return dictionary
        sample = {
            'data': sequence,
            'mask': mask
        }
        
        if self.return_participant_id:
            sample['participant_id'] = self.participant_ids[idx]
        
        return sample
    
    def get_feature_info(self) -> Dict[str, int]:
        """Get information about the dataset features."""
        if len(self.sequences) == 0:
            return {}
        
        sample_sequence = self.sequences[0]
        return {
            'sequence_length': sample_sequence.shape[0],
            'n_features': sample_sequence.shape[1],
            'total_sequences': len(self.sequences),
            'n_participants': len(set(self.participant_ids))
        }

class PhysiologicalDataLoader:
    """
    High-level wrapper for creating data loaders with sensible defaults.
    """
    
    def __init__(self, data_path: str, config: Optional[Dict] = None):
        """
        Initialize the data loader factory.
        
        Args:
            data_path: Path to the data directory
            config: Optional configuration dictionary
        """
        self.data_path = data_path
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'sequence_length': 24,
            'overlap': 0.2,
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': False,
            'drop_last': False,
            # point this at your JSON of "features to use"
            'selected_features_path': 'config/selected_features.json'
        }
        
        # Merge with user config
        self.config = {**self.default_config, **self.config}
    
    def create_dataloader(
        self,
        participants: List[str],
        data_type: str = 'train',
        filter_stress: bool = False,
        **kwargs
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for the specified participants.
        
        Args:
            participants: List of participant IDs
            data_type: 'train', 'val', or 'test'
            filter_stress: If True, only load stress-free (level 0) data.
            **kwargs: Additional arguments to override default config
            
        Returns:
            PyTorch DataLoader
        """
        # Override config with any provided kwargs
        config = {**self.config, **kwargs}
        
        # Create dataset
        dataset = PhysiologicalDataset(
            data_path=self.data_path,
            participants=participants,
            sequence_length=config['sequence_length'],
            overlap=config['overlap'],
            data_type=data_type,
            filter_stress=filter_stress, # Pass the flag here
            # hand off your feature‐list here:
            selected_features_path=config.get('selected_features_path')
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config['shuffle'] if data_type == 'train' else False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            drop_last=config['drop_last']
        )
        
        return dataloader

    def create_personalized_loaders(
        self,
        participant: str,
        data_params: Dict[str, Dict],
        filter_stress_train: bool = False,
        filter_stress_val: bool = False,
        filter_stress_test: bool = False,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, val, and test loaders for a single participant.
        
        Args:
            participant: Participant ID
            data_params: Dictionary with specific params for 'train', 'val', 'test'
            filter_stress_train: Whether to filter stress for the training set.
            filter_stress_val: Whether to filter stress for the validation set.
            filter_stress_test: Whether to filter stress for the test set.
            **kwargs: Additional configuration
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_params = {**data_params.get('train', {}), **kwargs}
        val_params = {**data_params.get('val', {}), **kwargs}
        test_params = {**data_params.get('test', {}), **kwargs}

        train_loader = self.create_dataloader(
            participants=[participant],
            data_type='train',
            filter_stress=filter_stress_train,
            **train_params
        )
        val_loader = self.create_dataloader(
            participants=[participant],
            data_type='val',
            shuffle=False,  # Never shuffle val data
            filter_stress=filter_stress_val,
            **val_params
        )
        test_loader = self.create_dataloader(
            participants=[participant],
            data_type='test',
            shuffle=False,  # Never shuffle test data
            filter_stress=filter_stress_test,
            **test_params
        )
        return train_loader, val_loader, test_loader

    def create_general_loaders(
        self,
        participants: List[str],
        data_params: Dict[str, Dict],
        filter_stress_train: bool = False,
        filter_stress_val: bool = False,
        filter_stress_test: bool = False,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, val, and test loaders for multiple participants.
        
        Args:
            participants: List of participant IDs
            data_params: Dictionary with specific params for 'train', 'val', 'test'
            filter_stress_train: Whether to filter stress for the training set.
            filter_stress_val: Whether to filter stress for the validation set.
            filter_stress_test: Whether to filter stress for the test set.
            **kwargs: Additional configuration
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_params = {**data_params.get('train', {}), **kwargs}
        val_params = {**data_params.get('val', {}), **kwargs}
        test_params = {**data_params.get('test', {}), **kwargs}

        train_loader = self.create_dataloader(
            participants=participants,
            data_type='train',
            filter_stress=filter_stress_train,
            **train_params
        )
        val_loader = self.create_dataloader(
            participants=participants,
            data_type='val',
            shuffle=False,  # Never shuffle val data
            filter_stress=filter_stress_val,
            **val_params
        )
        test_loader = self.create_dataloader(
            participants=participants,
            data_type='test',
            shuffle=False,  # Never shuffle test data
            filter_stress=filter_stress_test,
            **test_params
        )
        # Log summary of loaded sequences
        logger.debug("─" * 20 + " General Loaders Summary " + "─" * 20)
        train_sequences = len(train_loader.dataset)
        val_sequences = len(val_loader.dataset)
        test_sequences = len(test_loader.dataset)
        logger.debug(f"Train loader: {train_sequences} sequences loaded for {len(participants)} participants.")
        logger.debug(f"Val loader: {val_sequences} sequences loaded for {len(participants)} participants.")
        logger.debug(f"Test loader: {test_sequences} sequences loaded for {len(participants)} participants.")
        return train_loader, val_loader, test_loader
    

# Utility functions for data loading
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for handling batches.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Stack all tensors
    data = torch.stack([sample['data'] for sample in batch])
    mask = torch.stack([sample['mask'] for sample in batch])
    
    result = {
        'data': data,
        'mask': mask
    }
    
    # Add participant IDs if present
    if 'participant_id' in batch[0]:
        result['participant_id'] = [sample['participant_id'] for sample in batch]
    
    return result

def get_data_statistics(dataloader: DataLoader) -> Dict[str, float]:
    """
    Calculate statistics for the dataset.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        Dictionary with dataset statistics
    """
    total_values = 0
    total_valid_values = 0
    for batch in dataloader:
        data = batch['data']  # shape: (batch_size, sequence_length, n_features)
        mask = batch['mask']
        batch_size = data.size(0)
        sequence_length = data.size(1)
        n_features = data.size(2)
        total_values += batch_size * sequence_length * n_features
        total_valid_values += mask.sum().item()
    
    return {
        'total_values': total_values,
        'total_valid_values': total_valid_values,
        'missing_data_ratio': 1 - (total_valid_values / total_values)
    }

