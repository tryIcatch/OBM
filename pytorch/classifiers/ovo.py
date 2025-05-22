import time
import torch
import numpy as np
from tqdm import tqdm
from .base import BaseClassifier
from pytorch.models.svm import BinarySVM
from ..config import Config


class OVOClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.models = []
        self.pairs = []

    def fit(self, dataset):
        start_time = time.time()
        X, y = self._prepare_data(dataset)
        self.classes_ = np.unique(y)
        self.pairs = [(i, j) for i in self.classes_ for j in self.classes_ if i < j]

        for cls1, cls2 in tqdm(self.pairs, desc="Training OVO"):
            mask = np.isin(y, [cls1, cls2])
            X_pair = X[mask]
            y_binary = torch.where(torch.tensor(y[mask]) == cls1, 1.0, -1.0).to(Config.DEVICE)

            model = BinarySVM(X_pair.shape[1]).to(Config.DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE)

            prev_loss = float('inf')
            for _ in range(Config.MAX_ITER):
                optimizer.zero_grad()
                outputs = model(X_pair)
                loss = self._hinge_loss(outputs, y_binary)
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

    def predict(self, dataset):
        X, _ = self._prepare_data(dataset)
        votes = np.zeros((len(X), len(self.classes_)))

        for (cls1, cls2), model in zip(self.pairs, self.models):
            model = model.to(Config.DEVICE)
            with torch.no_grad():
                outputs = model(X)
            preds = np.where(outputs.cpu() > 0, cls1, cls2)
            preds_idx = np.searchsorted(self.classes_, preds)
            votes[np.arange(len(X)), preds_idx] += 1

        return self.classes_[np.argmax(votes, axis=1)]

