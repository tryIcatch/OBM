# OBM: Optimized Balanced SVM for Multiclass Classification

This repository provides the implementation of **OBM (Optimized Balanced Multiclass SVM)**, an efficient and accurate multiclass classification framework based on Support Vector Machines (SVM). OBM addresses the challenge of class imbalance and reduces the number of binary classifiers while preserving classification accuracy.

> 📌 **Paper Title**: OBM: An Optimized Balanced Support Vector Machine Multiclass Classification Scheme  
> 📍 **Conference**: International Conference on Information and Communications Security (ICICS)  
> 📎 **GitHub**: [https://github.com/tryIcatch/OBM.git](https://github.com/tryIcatch/OBM.git)

---

## 🔍 Highlights

- Balancing accuracy and efficiency in SVM-based multiclass classification
- A resampling-based strategy to balance positive and negative samples in each task
- Using no more than one binary classifier per class without sacrificing classification accuracy
- Improving recognition for minority classes in multiclass settings
- An efficient and accurate SVM multiclass scheme based on the above methods

---

## 📁 Project Structure



<pre>
<code>
pytorch/ 
├── classifiers/ # SVM-based classification strategies │ 
├── base.py # Base class for all SVM classifiers 
│ ├── obm.py # OBM method (proposed approach) 
│ ├── ovo.py # One-vs-One SVM implementation 
│ ├── ovo_smote.py # OVO with SMOTE for imbalance handling 
│ ├── ovr.py # One-vs-Rest SVM implementation 
│ ├── ovr_smote.py # OVR with SMOTE for imbalance handling 
│ ├── data/ # Data loading and preprocessing 
│ ├── datasets.py # Dataset configuration and splits 
│ ├── load_text.py # Utility for loading textual datasets 
│ ├── experiments/ # Scripts for experiment control
├── models/ # Directory to save trained model checkpoints 
├── results/ # Evaluation results, logs, and plots 
├── config.py # Global configuration file test.py # Entry point for testing and evaluation 
</code>
</pre>
````

---

## 🧪 How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
````

### 2. Run Experiments

```bash
cd pytorch
python run_experiment.py 
```

Supported methods:

* `obm`
* `ovo`
* `ovr`
* `ovo_smote`
* `ovr_smote`

Configure dataset paths and hyperparameters in `config.py`.

---

## 📊 Results

OBM achieves better performance than traditional OVO and OVR on imbalanced datasets by:

* Maintaining high accuracy on minority classes
* Reducing the number of binary classifiers from $\frac{K(K-1)}{2}$ to $K$
* Achieving a favorable trade-off between classification accuracy and training efficiency

---

## 🔗 Citation

If you find this project helpful, please consider citing our paper:

```bibtex
@inproceedings{shen2025obm,
  title={OBM: An Optimized Balanced Support Vector Machine Multiclass Classification Scheme},
  author={Shen, Hua et al.},
  booktitle={International Conference on Information and Communications Security (ICICS)},
  year={2025}
}
```

---

## 📬 Contact

For questions or collaborations, feel free to contact the corresponding author:

**Hua Shen**
