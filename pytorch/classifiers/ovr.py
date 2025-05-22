import time
import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from .base import BaseClassifier
from pytorch.models.svm import BinarySVM, BinarySVM_
from ..config import Config
import torch.nn as nn
from skimage.feature import hog


class OVRClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.models = []

    def fit(self, dataset):
        start_time = time.time()
        X, y = self._prepare_data(dataset)
        self.classes_ = np.unique(y)

        for cls_idx in tqdm(range(len(self.classes_)), desc="Training OVR"):
            y_binary = torch.where(torch.tensor(y) == cls_idx, 1.0, -1.0).to(Config.DEVICE)
            model = BinarySVM(X.shape[1]).to(Config.DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE)

            prev_loss = float('inf')
            for _ in range(Config.MAX_ITER):
                optimizer.zero_grad()
                outputs = model(X)
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
        scores = []
        for model in self.models:
            model = model.to(Config.DEVICE)
            with torch.no_grad():
                scores.append(model(X))
        return self.classes_[torch.stack(scores).argmax(dim=0).cpu().numpy()]

