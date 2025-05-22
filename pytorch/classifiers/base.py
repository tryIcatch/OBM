import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import time
from pytorch.config import Config
from torch.utils.data import DataLoader

class DataLoaderWrapper:
    @staticmethod
    def load_full_batch(dataset, device=Config.DEVICE):
        """加载数据并返回CPU标签"""
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        X, y = next(iter(loader))

        # 处理元组结构（如CIFAR10）
        if isinstance(X, (list, tuple)):
            X = X[0]

        # 特征送设备，标签保留在CPU
        return X.view(X.size(0), -1).to(device), y  # 标签保持CPU

    @staticmethod
    def create_loader(dataset, batch_size=256, shuffle=True):
        """创建标准数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )


class BaseClassifier:
    def __init__(self):
        self.classes_ = None
        self.train_time_ = 0.0

    def _prepare_data(self, dataset):
        X, y = DataLoaderWrapper.load_full_batch(dataset)
        if isinstance(X, (list, tuple)):
            X = X[0]
        X = X.view(X.size(0), -1).to(Config.DEVICE)
        return X, y.cpu().numpy()

    def _hinge_loss(self, outputs, y):
        return torch.clamp(1 - y * outputs, min=0).mean()

    def score(self, dataset):
        X, y = self._prepare_data(dataset)
        with torch.no_grad():
            preds = self.predict(dataset)
        return np.mean(preds == y)
