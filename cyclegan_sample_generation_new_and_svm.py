
import torch
import numpy as np
import random
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def to_numpy(data):
    """将 tensor 或 tensor 列表转换为 numpy 数组"""
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif isinstance(data, list):
        return np.array([to_numpy(x) for x in data])
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

def scalar_stand(Train_X, Test_X):
    # 用训练集标准差标准化训练集以及测试集
    scalar_train = preprocessing.StandardScaler().fit(Train_X)
    #scalar_test = preprocessing.StandardScaler().fit(Test_X)
    Train_X = scalar_train.transform(Train_X)
    Test_X = scalar_train.transform(Test_X)
    return Train_X, Test_X, scalar_train

def samlpe_generation_feed_svm(add_quantity,test_x,test_y,generator,domain_A_train_x, domain_A_train_y,domain_B_train_x,domain_B_train_y, c=0.2, g=0.001,return_model=False, excluded_fault_label=None, ):
    
    """
    Unified behavior:
      - excluded_fault_label=None => baseline (use/generate all 0..8)
      - excluded_fault_label=int  => leave-one-out training augmentation (skip that label in GAN generation),
                                   but test still includes it (so you can measure impact).
    """

    all_labels = list(range(9))
    if excluded_fault_label is None:
        fault_labels = all_labels
    else:
        if excluded_fault_label not in all_labels:
            raise ValueError(f"excluded_fault_label must be in 0..8, got {excluded_fault_label}")
        fault_labels = [l for l in all_labels if l != excluded_fault_label]

    # Ensure numpy
    test_X = to_numpy(test_x)
    test_Y = to_numpy(test_y).reshape(-1)

    
    # Baseline: no synthetic samples
    if add_quantity == 0:
        Generated_samples = np.empty((0, test_X.shape[1]), dtype=np.float32)
        Labels = np.empty((0,), dtype=np.int64)
    else:
        Generated_samples = []
        Labels = []

        print("Generating synthetic faults for labels:", fault_labels)

        for i in fault_labels:
            labels = np.full((add_quantity, 1), i, dtype=np.int64)
            indices_array = np.arange(add_quantity)

            domain_A_batch = torch.from_numpy(domain_A_train_x[indices_array]).float()
            labels_t = torch.from_numpy(labels).long()

            with torch.no_grad():
                fake_B = generator(domain_A_batch, labels_t)

            fake_B_np = fake_B.cpu().detach().numpy()
            Generated_samples.append(fake_B_np)
            Labels.append(labels.reshape(-1))

        Generated_samples = np.concatenate(Generated_samples, axis=0)
        Labels = np.concatenate(Labels, axis=0)

    domain_A_y = np.ravel(domain_A_train_y)
    domain_B_y = np.ravel(domain_B_train_y)

    # Build SVM train set: real B + synthetic B
    train_X = np.concatenate([domain_B_train_x, Generated_samples, domain_A_train_x], axis=0)
    train_Y = np.concatenate([domain_B_y.reshape(-1), Labels,domain_A_y], axis=0)

    # Standardize
    train_X, test_X_std, scaler_train = scalar_stand(train_X, test_X)

    # Train SVM
    classifier = SVC(C=c, gamma=g)

    print("Unique train labels:", np.unique(train_Y))
    print("Unique test labels:", np.unique(test_Y))

    classifier.fit(train_X, train_Y)

    # Predict
    Y_pred = classifier.predict(test_X_std)

    labels_all = sorted(set(np.unique(test_Y)).union(set(np.unique(train_Y))))
    cm = confusion_matrix(test_Y, Y_pred, labels=labels_all)
    acc_all = accuracy_score(test_Y, Y_pred)

    print("Labels used in confusion matrix:", labels_all)
    print(cm)
    print(f"Accuracy (ALL test classes): {acc_all:.6f}")

    # If excluded label exists in test: also report known-class accuracy
    acc_known = None
    if excluded_fault_label is not None:
        mask = (test_Y != excluded_fault_label)
        if np.any(mask):
            acc_known = accuracy_score(test_Y[mask], Y_pred[mask])
            print(f"Accuracy (KNOWN classes only, excluding {excluded_fault_label}): {acc_known:.6f}")

    if return_model:
        # Keep backward compatibility: return classifier + scaler + primary accuracy
        # Primary accuracy = ALL classes (so you always see the impact of reintroducing the unseen fault)
        return classifier, scaler_train, acc_all

    return acc_all
  
  


    
