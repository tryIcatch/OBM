import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


class CustomTextDataset(Dataset):
    """自定义文本数据集类，用于加载文本数据并将其转换为PyTorch可用的格式"""

    def __init__(self, X, y):
        """
        初始化自定义文本数据集
        Args:
            X (scipy.sparse.csr_matrix): 稀疏矩阵格式的特征数据（TF-IDF 特征）
            y (numpy.ndarray): 标签数据
        """
        self.X = X
        self.y = y

    def __len__(self):
        """返回数据集的大小"""
        return self.X.shape[0]  # 使用稀疏矩阵的第一维大小（即样本数量）

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        Args:
            idx (int): 索引
        Returns:
            tuple: (特征, 标签)，特征是一个 PyTorch Tensor，标签是一个整数
        """
        # 获取对应的特征和标签
        sample = self.X[idx]
        label = self.y[idx]

        # 将稀疏矩阵转换为 dense 格式的 Tensor
        feature = torch.tensor(sample.toarray(), dtype=torch.float32).squeeze(0)

        return feature, label
