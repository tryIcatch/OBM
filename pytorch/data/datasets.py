import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, Subset
from pytorch.config import Config

def get_transform(dataset_name):
    if dataset_name in ['mnist', 'fashion']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name == 'digits':
        return transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize(28),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name in ['Letter', 'iris', 'covtype', 'wine', 'breast_cancer', 'yeast', 'glass', 'wine_quality', 'kdd', 'nsl-kdd']:
        return None
    raise ValueError(f"Unsupported dataset: {dataset_name}")



def _load_dataset(name, train=None, sample_size=None, split_type='iid', imbalance_factor=None):
    transform = get_transform(name)
    if name == 'mnist':
        dataset = datasets.MNIST(Config.DATA_ROOT, train=train, download=True, transform=transform)
    elif name == 'fashion':
        dataset = datasets.FashionMNIST(Config.DATA_ROOT, train=train, download=True, transform=transform)
    elif name == 'digits':
        digits = load_digits()
        X, y = digits.data, digits.target
        X = X.reshape(-1, 8, 8).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_train, X_test = X_train / 16.0, X_test / 16.0
        if train:
            dataset = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train))
        else:
            dataset = TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test))
        if sample_size and sample_size < len(dataset):
            indices = torch.randperm(len(dataset))[:sample_size].tolist()
            dataset = Subset(dataset, indices)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    if sample_size and len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        dataset = Subset(dataset, indices)
    if split_type == 'no_iid':
        dataset = create_non_iid_split(dataset, num_classes=10)
    elif split_type == 'imbalance':
        origin_data = dataset
        dataset = create_imbalanced_data(dataset, imbalance_factor)
        plot_dual_axis_comparison(origin_data, dataset, name)
    return dataset

def create_non_iid_split(dataset, num_classes, samples_per_class=500):
    indices = []
    class_counts = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_counts[label].append(idx)
    for cls in range(0, num_classes, 2):
        if not class_counts[cls] or not class_counts[(cls + 1) % num_classes]:
            continue
        max_samples = min(len(class_counts[cls]), len(class_counts[(cls + 1) % num_classes]))
        actual_samples_per_class = min(samples_per_class, max_samples)
        cls1_indices = np.random.choice(class_counts[cls], actual_samples_per_class, replace=False)
        cls2_indices = np.random.choice(class_counts[(cls + 1) % num_classes], actual_samples_per_class, replace=False)
        indices.extend(cls1_indices.tolist() + cls2_indices.tolist())
    assert max(indices) < len(dataset)
    return Subset(dataset, indices)

def create_imbalanced_data(dataset, imbalance_factor):
    class_counts = get_class_distribution(dataset)
    classes = sorted(class_counts.keys())
    total_samples = sum(class_counts.values())
    class_weights = [1.0 / (imbalance_factor ** (i / (len(classes) - 1))) for i in range(len(classes))]
    sum_weights = sum(class_weights)
    class_proportions = [w / sum_weights for w in class_weights]
    class_samples = [min(class_counts[cls], int(prop * total_samples)) for cls, prop in zip(classes, class_proportions)]
    indices = []
    for cls, n in zip(classes, class_samples):
        cls_indices = [i for i, (_, label) in enumerate(dataset) if label == cls]
        indices.extend(np.random.choice(cls_indices, n, replace=False))
    return Subset(dataset, indices)

def get_class_distribution(dataset):
    if isinstance(dataset, Subset):
        labels = [dataset.dataset[i][1] for i in dataset.indices]
    else:
        labels = [item[1] for item in dataset]
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

def plot_dual_axis_comparison(original_dataset, subset, name, class_names=None):
    def safe_style():
        available = plt.style.available
        if 'seaborn-whitegrid' in available:
            return 'seaborn-whitegrid'
        elif 'seaborn-v0_8-whitegrid' in available:
            return 'seaborn-v0_8-whitegrid'
        return 'ggplot'

    COLOR_PALETTE = {
        'original': '#417EBD',
        'subset': '#E85555',
        'grid': '#D0D0D0',
        'text': '#404040',
        'background': '#FFFFFF'
    }

    plt.style.use(safe_style())
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.facecolor': COLOR_PALETTE['background'],
        'axes.edgecolor': COLOR_PALETTE['text'],
        'axes.labelcolor': COLOR_PALETTE['text'],
        'xtick.color': COLOR_PALETTE['text'],
        'ytick.color': COLOR_PALETTE['text'],
        'grid.color': COLOR_PALETTE['grid'],
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })

    def get_labels(dataset):
        if isinstance(dataset, Subset):
            return get_labels(dataset.dataset)[dataset.indices]
        elif hasattr(dataset, 'tensors') and len(dataset.tensors) > 1:
            return dataset.tensors[1].numpy()
        return np.array([label for _, label in dataset])

    orig_counts = np.bincount(get_labels(original_dataset))
    subset_counts = np.bincount(get_labels(subset), minlength=len(orig_counts))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(orig_counts))]

    def draw_bar(counts, color, name):
        x = np.arange(len(counts))
        fig, ax = plt.subplots(figsize=(14, 7))
        rects = ax.bar(x, counts, 0.4, color=color, edgecolor='white', linewidth=1.5, alpha=0.95)
        ax.set_ylabel("Sample Count", color=color, fontsize=13, labelpad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontstyle='italic', color=COLOR_PALETTE['text'])
        ax.spines['left'].set_color(color)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', colors=color)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height * 1.02, f'{height:,}', ha='center', va='bottom', color=COLOR_PALETTE['text'], fontsize=10)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(f'{name}_{color}.pdf', dpi=600)
        plt.close(fig)

    draw_bar(orig_counts, COLOR_PALETTE['original'], f'{name}_original')
    draw_bar(subset_counts, COLOR_PALETTE['subset'], f'{name}_subset')
