import torch
import numpy as np
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
import time
from tqdm import tqdm
from pytorch.classifiers.base import BaseClassifier
from pytorch.config import Config
from pytorch.models.svm import BinarySVM


class OVOSMOTEClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.models = []
        self.pairs = []

    def fit(self, dataset):
        start_time = time.time()
        X, y = self._prepare_data(dataset)
        self.classes_ = np.unique(y)
        self.pairs = [(i, j) for i in self.classes_ for j in self.classes_ if i < j]

        for cls1, cls2 in tqdm(self.pairs, desc="Training OVO with SMOTE"):
            mask = np.isin(y, [cls1, cls2])
            X_pair = X[mask].cpu().numpy()
            y_binary = np.where(y[mask] == cls1, 1.0, -1.0)

            minority_class_count = np.sum(y_binary == 1)
            if minority_class_count > 1:
                k_neighbors = min(5, minority_class_count - 1)
                smote = SMOTE(k_neighbors=k_neighbors)
                try:
                    X_resampled, y_resampled = smote.fit_resample(X_pair, y_binary)
                except ValueError:
                    X_resampled, y_resampled = X_pair, y_binary
            else:
                X_resampled, y_resampled = X_pair, y_binary

            X_resampled = torch.tensor(X_resampled, dtype=torch.float32).to(Config.DEVICE)
            y_resampled = torch.tensor(y_resampled, dtype=torch.float32).to(Config.DEVICE)

            model = BinarySVM(X_resampled.shape[1]).to(Config.DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE)

            prev_loss = float('inf')
            for _ in range(Config.MAX_ITER):
                optimizer.zero_grad()
                outputs = model(X_resampled)
                loss = self._hinge_loss(outputs, y_resampled)
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
