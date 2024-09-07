import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, root: str, subset: str, audio_dataset: str):
        """
        Args:
            root (str): Root directory of the dataset.
            subset (str): Subset of the dataset ('train', 'val', 'test').
            audio_dataset (str): Directory of the audio dataset.
        """
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.audio_dataset = audio_dataset

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class ESC50(AudioDataset):
    base_folder = 'recorded/'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta', 'esc50.csv'),
    }

    def __init__(self, root, subset, audio_dataset, reading_transformations: nn.Module = None, shot: int = 1, seed: int = 10):
        """
        Args:
            root (str): Root directory of the dataset.
            subset (str): Subset of the dataset ('train', 'val', 'test').
            audio_dataset (str): Directory of the audio dataset.
            reading_transformations (nn.Module, optional): Transformations to apply to the audio data.
            shot (int, optional): Number of samples per class for few-shot training. -1 means using all samples.
            seed (int, optional): Random seed for reproducibility.
        """
        super(ESC50, self).__init__(root, subset, audio_dataset)
        self.audio_dir = audio_dataset
        self.json_file = os.path.join(self.root, f"few-shot-json-seed{seed}", f"{audio_dataset}_{shot}_{subset}_data.json")
        print(self.audio_dir)
        self.pre_transformations = reading_transformations
        self.shot = shot
        self.seed = seed
        self.targets, self.audio_paths = [], []

        if not os.path.exists(os.path.join(self.root, f"few-shot-json-seed{seed}")):
            os.makedirs(os.path.join(self.root, f"few-shot-json-seed{seed}"))

        if os.path.exists(self.json_file):
            self._load_from_json()
            print('Load data from json...')
        else:
            self._load_meta()
            self._split_data(subset)
            self._save_to_json()
        
        self.class_to_idx = {category: i for i, category in enumerate(self.classes)}

    def _load_meta(self):
        """Load metadata from the CSV file."""
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path)
        self.classes = sorted(self.df[self.label_col].unique())

    def _split_data(self, subset):
        """Split the dataset into train, val, and test subsets."""
        
        if subset not in ['train', 'val', 'test']:
            raise ValueError("Subset must be one of 'train', 'val', or 'test'")

        split_ratio = [0.7, 0.2, 0.1]  # Train, Val, Test split ratio

        subset_df = pd.DataFrame(columns=self.df.columns)
        
        for category in self.classes:
            category_df = self.df[self.df[self.label_col] == category]
            category_df = category_df.sort_values(by='filename').reset_index(drop=True)  # Sort by filename
            train_end = int(split_ratio[0] * len(category_df))
            val_end = train_end + int(split_ratio[1] * len(category_df))
            
            if subset == 'train':
                if self.shot > 0:
                    np.random.seed(self.seed)
                    sampled_df = category_df.iloc[:train_end].sample(n=self.shot, random_state=self.seed)
                    subset_df = pd.concat([subset_df, sampled_df], ignore_index=True)
                else:
                    subset_df = pd.concat([subset_df, category_df.iloc[:train_end]], ignore_index=True)
            elif subset == 'val':
                subset_df = pd.concat([subset_df, category_df.iloc[train_end:val_end]], ignore_index=True)
            else:  # test
                subset_df = pd.concat([subset_df, category_df.iloc[val_end:]], ignore_index=True)

        for _, row in tqdm(subset_df.iterrows(), total=subset_df.shape[0]):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

    def _save_to_json(self):
        """Save the dataset information to a JSON file."""
        data = {
            "audio_paths": self.audio_paths,
            "targets": self.targets,
            "classes": self.classes
        }
        with open(self.json_file, 'w') as f:
            json.dump(data, f)

    def _load_from_json(self):
        """Load the dataset information from a JSON file."""
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            self.audio_paths = data['audio_paths']
            self.targets = data['targets']
            self.classes = data['classes']

    def __getitem__(self, index):
        """Get the item at the given index."""
        file_path, target = self.audio_paths[index], self.targets[index]
        try:
            idx = torch.tensor(self.class_to_idx[target])
        except KeyError:
            raise KeyError(f"Target '{target}' not found in class_to_idx dictionary.")
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        return file_path, target, one_hot_target

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.audio_paths)