import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import torch
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(self, df, non_feature_columns=None):
        self.non_feature_columns = non_feature_columns
        self.meta_data = df[non_feature_columns] if non_feature_columns else None
        self.features = df.loc[:, ~df.columns.isin(non_feature_columns)] if non_feature_columns else df
        self.features = self.features.fillna(self.features.mean())  
        self.features = (self.features-self.features.mean()) / self.features.std() # normalize

    def __len__(self):
        return len(self.features.index)

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[[idx]].values, dtype=torch.float64), self.meta_data.iloc[[idx]].values.flatten().tolist() # {k:list(v.values())[0] for k,v in self.meta_data.iloc[[idx]].to_dict().items()}
    
    def get_number_of_features(self):
        return len(self.features.columns)
