import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pickle
import torchaudio
from collections import defaultdict
# import random  # 导入random库

# 定义类别字典
classes = {
    'Background': 0,
    'Starter Gun': 1,
    'Door Slam': 2,
    'Car Alarm': 3,
    'Crackers': 4,
    'Cannon': 5,
    'Fountain Cannon': 6,
    'High Altitude Firework': 7
}

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

class Fiber(AudioDataset):
    def __init__(self, root, subset, audio_dataset, shot: int = -1, seed: int = 1):
        """
        Args:
            root (str): Root directory of the dataset.
            subset (str): Subset of the dataset ('train', 'val', 'test').
            audio_dataset (str): Directory of the audio dataset.
            shot (int, optional): Number of samples per class for few-shot training. -1 means using all samples.
            seed (int, optional): Random seed for reproducibility.
        """
        super(Fiber, self).__init__(root, subset, audio_dataset)
        self.data_path = root
        self.audio_dir = audio_dataset
        self.json_file = os.path.join(self.data_path, f"few-shot-json-seed{seed}", f"{audio_dataset}_{shot}_{subset}_data.json")
        self.shot = shot
        self.seed = seed
        self.targets, self.audio_paths = [], []

        if not os.path.exists(os.path.join(self.data_path, f"few-shot-json-seed{seed}")):
            os.makedirs(os.path.join(self.data_path, f"few-shot-json-seed{seed}"))

        if os.path.exists(self.json_file):
            self._load_from_json()
            print('Load data from json...')
        else:
            self._load_meta()
            if subset == 'train' or subset == 'val':
                self._split_train_val()
                if subset == 'train' and self.shot > 0:
                    self._apply_few_shot()
            else:
                self._split_data(subset)
            self._save_to_json()

        self.class_to_idx = classes
    def _load_meta(self):
        """Load metadata from the pickle file or from text files if available."""
        train_txt_file = os.path.join(self.data_path, self.audio_dir + '_train_labels.txt')
        test_txt_file = os.path.join(self.data_path, self.audio_dir + '_test_labels.txt')

        self._load_from_txt(train_txt_file, test_txt_file)
        print(f'Loaded labels from {train_txt_file} and {test_txt_file}')

    def _load_from_txt(self, train_txt_file, test_txt_file):
        """Load labels and classes from text files."""
        self.target_train = np.loadtxt(train_txt_file, dtype=int, delimiter=',', skiprows=1, usecols=1).tolist()
        self.target_test = np.loadtxt(test_txt_file, dtype=int, delimiter=',', skiprows=1, usecols=1).tolist()
        self.classes = list(classes.keys())

    def _split_train_val(self):
        """Split the train set into balanced train and val subsets."""
        class_samples = defaultdict(list)

        for index in range(len(self.target_train)):
            class_samples[self.target_train[index]].append(index)

        train_indices = []
        val_indices = []

        for class_id, indices in class_samples.items():
            indices.sort()
            split_idx = int(0.8 * len(indices))
            train_indices.extend(indices[:split_idx])
            val_indices.extend(indices[split_idx:])

        if self.subset == 'train':
            selected_indices = train_indices
        else:
            selected_indices = val_indices

        for index in tqdm(selected_indices):
            file_path = os.path.join(self.data_path, self.audio_dir, f'train_{index}.wav')
            self.targets.append(self.classes[self.target_train[index]])
            self.audio_paths.append(file_path)

    def _split_data(self, subset):
        """Split the dataset into train, val, and test subsets."""
        if subset == 'test':
            for index in tqdm(range(len(self.target_test))):
                file_path = os.path.join(self.data_path, self.audio_dir, f'test_{index}.wav')
                self.targets.append(self.classes[self.target_test[index]])
                self.audio_paths.append(file_path)

    def _apply_few_shot(self):
        """Apply few-shot sampling to the train set."""
        class_samples = defaultdict(list)

        for index in range(len(self.targets)):
            class_samples[self.targets[index]].append((index, self.audio_paths[index]))

        new_targets = []
        new_audio_paths = []

        for class_id, samples in class_samples.items():
            samples.sort(key=lambda x: x[1])  # 按文件名升序排序
            selected_samples = samples[:self.shot]  # 选择前shot个样本
            for index, file_path in selected_samples:
                new_targets.append(class_id)
                new_audio_paths.append(file_path)

        self.targets = new_targets
        self.audio_paths = new_audio_paths

    def _save_to_json(self):
        """Save the dataset information to a JSON file."""
        data = {
            "audio_paths": self.audio_paths,
            "targets": self.targets,
            "classes": list(classes.keys())
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
        one_hot_target = torch.zeros(len(self.class_to_idx)).scatter_(0, idx, 1).reshape(1, -1)
        return file_path, target, one_hot_target

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.audio_paths)
