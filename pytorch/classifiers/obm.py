import time
import torch
import numpy as np
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm
from sklearn.utils import resample
from .base import BaseClassifier
from pytorch.models.svm import BinarySVM, BinarySVM_
from ..config import Config
from torchvision import datasets, transforms, models

class OBMClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.models = []

    def fit(self, dataset):

        start_time = time.time()
        X, y = self._prepare_data(dataset)
        X_np, y_np = X.cpu().numpy(), y

        self.classes_ = np.unique(y_np)

        for cls_idx in tqdm(range(len(self.classes_)), desc="Training OBM"):
            # 获取正样本
            pos_mask = (y_np == cls_idx)
            X_pos = X_np[pos_mask]
            n_pos = len(X_pos)

            if n_pos == 0:
                self.models.append(None)
                continue


            X_neg = self._sample_negative(X_np, y_np, cls_idx, n_pos)
            if len(X_neg) == 0:
                self.models.append(None)
                continue


            X_train = np.vstack([X_pos, X_neg])
            y_train = np.hstack([np.ones(n_pos), -np.ones(len(X_neg))])


            X_tensor = torch.tensor(X_train, dtype=torch.float32).to(Config.DEVICE)
            y_tensor = torch.tensor(y_train, dtype=torch.float32).to(Config.DEVICE)


            input_dim = X_tensor.shape[1]
            model = BinarySVM(input_dim).to(Config.DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE)

            prev_loss = float('inf')
            for _ in range(Config.MAX_ITER):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = self._hinge_loss(outputs, y_tensor)
                loss.backward()
                optimizer.step()

                if abs(loss.item() - prev_loss) < Config.TOLERANCE:
                    break
                prev_loss = loss.item()
                del outputs, loss
                torch.cuda.empty_cache()
            self.models.append(model.cpu())

        self.train_time_ = time.time() - start_time
        return self

    def _sample_negative(self, X, y, pos_class, n_pos):

        neg_mask = (y != pos_class)
        X_neg = X[neg_mask]
        y_neg = y[neg_mask]

        classes, counts = np.unique(y_neg, return_counts=True)
        proportions = counts / counts.sum()
        samples_per_class = (proportions * n_pos).astype(int)

        sampled = []
        for cls, n in zip(classes, samples_per_class):
            mask = (y_neg == cls)
            available = mask.sum()
            if available == 0 or n == 0:
                continue
            replace = n > available
            sampled.append(resample(X_neg[mask], n_samples=n, replace=replace))

        return np.vstack(sampled) if sampled else np.empty((0, X.shape[1]))

    def predict(self, dataset):
        X, _ = self._prepare_data(dataset)
        scores = []

        for model in self.models:
            if model is None:
                scores.append(torch.full((len(X),), -np.inf, device=Config.DEVICE))
            else:
                model = model.to(Config.DEVICE)
                with torch.no_grad():
                    scores.append(model(X))

        return self.classes_[torch.stack(scores).argmax(dim=0).cpu().numpy()]

