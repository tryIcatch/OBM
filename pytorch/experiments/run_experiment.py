import time
from pytorch.classifiers.ovo import OVOClassifier
from pytorch.classifiers.ovo_smote import OVOSMOTEClassifier
from pytorch.classifiers.ovr import OVRClassifier
from pytorch.classifiers.obm import OBMClassifier
from pytorch.classifiers.ovr_smote import OVRSMOTEClassifier
from pytorch.config import Config
from pytorch.data.datasets import _load_dataset


def print_results(results, dataset_name):
    print(f"\n{dataset_name.upper()} Results:")
    print("{:<12} | {:<12} | {:<15} | {:<15}".format(
        "Classifier", "Train Time", "Train Accuracy", "Test Accuracy"
    ))
    print("-" * 60)
    for res in results:
        print("{:<12} | {:<12.2f} | {:<15.2f} | {:<15.2f}".format(
            res['classifier'],
            res['train_time'],
            res['train_acc'],
            res['test_acc']
        ))



def run_experiment(dataset_name, split_type='iid'):
    imbalance_factor = 200 if split_type == 'imbalance' else None

    if dataset_name in ['STL10', 'SVHN']:
        train_set = _load_dataset(
            dataset_name,
            train="train",
            sample_size=Config.SAMPLE_SIZE,
            split_type=split_type,
            imbalance_factor=imbalance_factor
        )
        test_set = _load_dataset(
            dataset_name,
            train="test",
            sample_size=Config.TEST_SIZE
        )
    else:
        train_set = _load_dataset(
            dataset_name,
            train=True,
            sample_size=Config.SAMPLE_SIZE,
            split_type=split_type,
            imbalance_factor=imbalance_factor
        )
        test_set = _load_dataset(
            dataset_name,
            train=False,
            sample_size=Config.TEST_SIZE
        )


        classifiers = {
            'OVO': OVOClassifier(),
            'OVR': OVRClassifier(),
            'OBM': OBMClassifier(),
            'OVOSMOTE': OVOSMOTEClassifier(),
            'OVRSMOTE': OVRSMOTEClassifier()
        }

    print(len(train_set))

    results = []
    for name, clf in classifiers.items():
        print(f"\n[{split_type.upper()}] training {name} classifiers ...")
        start_time = time.time()
        clf.fit(train_set)
        train_acc = clf.score(train_set)
        test_acc = clf.score(test_set)
        results.append({
            'dataset': dataset_name,
            'split_type': split_type,
            'classifier': name,
            'train_time': clf.train_time_,
            'train_acc': train_acc,
            'test_acc': test_acc
        })

    print_results(results, f"{dataset_name} ({split_type})")
    return results


if __name__ == "__main__":
    for dataset in ['digits', 'fashion', 'mnist']:
        for split_type in ['imbalance', 'iid']:
            run_experiment(dataset, split_type=split_type)
