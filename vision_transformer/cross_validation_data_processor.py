import os
import torch
import random
import shutil
from sklearn.model_selection import KFold
from torchvision import datasets, transforms

class CrossValidationDataset:
    def __init__(self, data_dir, k_folds, output_dir):
        self.data_dir = data_dir
        self.k_folds = k_folds
        self.output_dir = output_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        self.kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    def create_folds(self):
        os.makedirs(self.output_dir, exist_ok=True)
        indices = list(range(len(self.dataset)))
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(indices)):
            fold_dir = os.path.join(self.output_dir, f'fold_{fold}')
            train_dir = os.path.join(fold_dir, 'train')
            val_dir = os.path.join(fold_dir, 'val')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            self._copy_data(train_idx, train_dir)
            self._copy_data(val_idx, val_dir)

    def _copy_data(self, indices, target_dir):
        for idx in indices:
            src_path, label = self.dataset.samples[idx]
            label_dir = os.path.join(target_dir, self.dataset.classes[label])
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(src_path, label_dir)

def main():
    data_processor = CrossValidationDataset('./data', k_folds=5, output_dir='./cross_validation_data')
    data_processor.create_folds()

if __name__ == "__main__":
    main()
