from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import pandas as pd
import os
import torch.nn as nn
import torch

class AudioDataset(Dataset):
    def __init__(self, root: str, test_file:str):
        self.root = os.path.expanduser(root)
        self.test_file = test_file

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class ESC50(AudioDataset):
    base_folder = 'recorded/'
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    filename = "ESC-50-master.zip"
    num_files_in_dir = 2000
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta', 'esc50.csv'),
    }

    def __init__(self, root, test_file, reading_transformations: nn.Module = None):
        super(ESC50, self).__init__(root, test_file)
        self._load_meta()
        self.audio_dir = test_file
        print(self.audio_dir)
        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        self.df['category'] = self.df['category'].str.replace('_', ' ')

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_', ' ') for x in sorted(self.df[self.label_col].unique())]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)
