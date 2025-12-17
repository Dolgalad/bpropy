import os
import shutil
import pickle as pkl
import torch
from torch_geometric.data import Data, Dataset, batch
from torch_geometric.loader.dataloader import Collater

class MRFDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, in_memory=True, limit=None):
        self.in_memory = in_memory
        super().__init__(root, transform, pre_transform, pre_filter)
        if os.path.exists(os.path.join(root, "processed")):
            shutil.rmtree(os.path.join(root, "processed"))
        if os.path.exists(os.path.join(root, "raw")):
            shutil.rmtree(os.path.join(root, "raw"))
        self.data = None
        if self.in_memory:
            with open(root, "rb") as f:
                self.data = pkl.load(f)
                if limit is not None:
                    self.data = self.data[:limit]


    @property
    def raw_file_names(self):
        if self.in_memory:
            return [self.root]
        return sorted([f for f in os.listdir(self.root) if f!="processed" and f!="raw"], key=lambda p: int(p.split(".")[0]))
    @property
    def processed_file_names(self):
        return self.raw_file_names
    #def download(self):
    #    pass
    #def process(self):
    #    pass
    def len(self):
        if self.in_memory:
            return len(self.data)
        return len(self.processed_file_names)
    def get(self, idx):
        if self.in_memory:
            return self.data[idx]
        else:
            with open(os.path.join(self.root, self.processed_file_names[idx]),"rb") as f:
                return pkl.load(f)
            #return torch.load(os.path.join(self.root, self.processed_file_names[idx]))
