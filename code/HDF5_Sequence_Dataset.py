import torch
import tables
import numpy as np
from typing import Optional, Any

class HDF5SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, sequence_key: str = 'sequences', target_key: str = 'targets', 
                 transform: Optional[Any] = None, start: int = 0, stop: Optional[int] = None,
                 exclude_abs_close: bool = True):
        self.h5_path = h5_path
        self.sequence_key = sequence_key
        self.target_key = target_key
        self.transform = transform
        self.h5: Optional[tables.File] = None  # Lazy loading - don't open immediately
        self.sequences: Optional[tables.EArray] = None
        self.targets: Optional[tables.EArray] = None
        self.start = start
        self.stop = stop
        self.length: Optional[int] = None
        self.exclude_abs_close = exclude_abs_close
        self._initialize_lazy()

    def _initialize_lazy(self):
        """Initialize file access only when needed to get dataset size"""
        if self.h5 is None:
            self.h5 = tables.open_file(self.h5_path, mode='r')
            self.sequences = self.h5.get_node(f'/{self.sequence_key}')  # type: ignore
            self.targets = self.h5.get_node(f'/{self.target_key}')      # type: ignore
        
        if self.stop is None:
            self.stop = self.sequences.shape[0]  # type: ignore
        self.length = self.stop - self.start

    def _ensure_file_open(self):
        """Ensure file is open, reopen if needed (for multiprocessing)"""
        if self.h5 is None or not self.h5.isopen:
            self.h5 = tables.open_file(self.h5_path, mode='r')
            self.sequences = self.h5.get_node(f'/{self.sequence_key}')  # type: ignore
            self.targets = self.h5.get_node(f'/{self.target_key}')      # type: ignore

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Ensure file is open and accessible
        self._ensure_file_open()
        
        real_idx = self.start + idx
        # PyTables EArray supports direct indexing - this only loads the specific slice
        # Direct indexing is memory efficient and only loads what we need
        seq = torch.from_numpy(self.sequences[real_idx]).float()  # type: ignore
        tgt = torch.from_numpy(self.targets[real_idx]).float()    # type: ignore
        
        # If exclude_abs_close is True, remove the last feature (index 6: absolute final close)
        if self.exclude_abs_close:
            seq = seq[:, :, :-1]  # Remove last feature dimension
        
        if self.transform:
            seq, tgt = self.transform(seq, tgt)
            
        return seq, tgt

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None
            self.sequences = None
            self.targets = None

    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass

# # Example usage:
# dataset = HDF5SequenceDataset('0_to_299940_stride_60_seqlen_1200.h5')
# loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
# for x, y in loader:
#     print(x.shape, y.shape)
# dataset.close()
