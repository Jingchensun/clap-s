import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import torchaudio
from collections import defaultdict
from tqdm import tqdm

# 定义类别字典
classes = {
    'diver_5knock': 1,
    'diver_shake': 2,
    'pnumetric_drill': 3,
    'swazall_on_cable': 4
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
                
        # 显式赋值类列表
        self.classes = list(classes.keys())  # 确保 `self.classes` 正确初始化

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

        self.class_to_idx = {v: k for k, v in classes.items()}  # 反转 classes 映射，使 ID 对应类别名称

    def _load_meta(self):
        """Load metadata from the text files."""
        train_txt_file = os.path.join(self.data_path, 'train.txt')
        test_txt_file = os.path.join(self.data_path, 'test.txt')

        self._load_from_txt(train_txt_file, test_txt_file)
        print(f'Loaded labels from {train_txt_file} and {test_txt_file}')

    def _load_from_txt(self, train_txt_file, test_txt_file):
        """Load labels and classes from text files."""
        self.target_train, self.audio_paths_train = [], []
        self.target_test, self.audio_paths_test = [], []

        # 解析 train.txt
        if os.path.exists(train_txt_file):
            with open(train_txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(', ')
                    if len(parts) == 3:
                        audio_path, class_id, class_name = parts
                        class_id = int(class_id)  # 转换为整数
                        self.target_train.append(class_id)
                        self.audio_paths_train.append(os.path.join(self.data_path, self.audio_dir, audio_path))

        # 解析 test.txt（如果有）
        if os.path.exists(test_txt_file):
            with open(test_txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(', ')
                    if len(parts) == 3:
                        audio_path, class_id, class_name = parts
                        class_id = int(class_id)  # 转换为整数
                        self.target_test.append(class_id)
                        self.audio_paths_test.append(os.path.join(self.data_path, self.audio_dir, audio_path))

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
            self.targets.append(self.target_train[index])
            self.audio_paths.append(self.audio_paths_train[index])

    def _split_data(self, subset):
        """Split the dataset into test subset."""
        if subset == 'test':
            for index in tqdm(range(len(self.target_test))):
                self.targets.append(self.target_test[index])
                self.audio_paths.append(self.audio_paths_test[index])

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
            for _, file_path in selected_samples:
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
            class_idx = target - 1  # 类别 ID 转换为索引
            one_hot_target = torch.zeros(len(self.class_to_idx)).scatter_(0, torch.tensor(class_idx), 1)
        except KeyError:
            raise KeyError(f"Target '{target}' not found in class_to_idx dictionary.")
        return file_path, target, one_hot_target

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.audio_paths)
