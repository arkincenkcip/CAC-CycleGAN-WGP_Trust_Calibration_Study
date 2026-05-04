"""
Load Model and Test Script
加载训练好的 CycleGAN 模型并进行测试

功能：
1. 加载保存的最佳模型
2. 使用 g_AB 从健康样本生成故障样本
3. 使用 g_BA 从故障样本生成健康样本
4. 调用 SVM 评估函数进行诊断
5. 可视化生成结果
"""

import torch
import numpy as np
import os
from CAC_CycleGAN_WGP_pytorch import CycleGAN




# --- Portable path setup ---
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
OUT_DIR = PROJECT_ROOT / "resources" / "models" /"epoch_models" /"generators_all_faults"
#OUT_DIR = PROJECT_ROOT / "resources" / "models" /"epoch_models" /"generators_no_2_fault"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- end portable path setup ---

#MODEL_PATH = OUT_DIR/'model_acc_100.00_epoch_756.pth'
MODEL_PATH = OUT_DIR/'model_acc_100.00_epoch_291.pth'




DATA_PATH = DATA_DIR / "ALL_TEST_SETS_COMBINED.npz"




class ModelTester:
    """模型测试类"""
    
    def __init__(self, model_path, data_path, device='cuda', excluded_fault_label=None, ref_data_path=None): #SET EXCLUDED LABEL HERE!!!

        """
        初始化模型测试器

        Args:
            model_path: 模型文件路径
            data_path: 数据文件路径
            device: 运行设备 ('cuda' or 'cpu')
            ref_data_path: optional separate file to load S2 training references from;
                           if None, training references are loaded from data_path
        """

        self.excluded_fault_label = excluded_fault_label
        print(f"Excluded fault:{self.excluded_fault_label}")

        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # 初始化模型
        print("\nInitializing CycleGAN model...")
        self.gan = CycleGAN(device=self.device)

        # 加载模型权重
        print(f"Loading model from: {model_path}")
        self.load_model(model_path)

        # 加载数据
        effective_ref_path = ref_data_path if ref_data_path is not None else data_path
        print(f"Loading data from: {data_path}")
        print(f"Loading S2 references from: {effective_ref_path}")
        self.load_data(data_path, ref_data_path=effective_ref_path)


        print("\n✓ Model and data loaded successfully!\n")
    
    def load_model(self, model_path):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print("STEP1: starting torch.load")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        print("STEP2: checkpoint loaded")
        # 加载生成器和判别器权重
        self.gan.g_AB.load_state_dict(checkpoint['g_AB_state_dict'])
        self.gan.g_BA.load_state_dict(checkpoint['g_BA_state_dict'])

        
        # 设置为评估模式
        self.gan.g_AB.eval()
        self.gan.g_BA.eval()

        
        # 打印模型信息
        if 'epoch' in checkpoint:
            print(f"  Model from epoch: {checkpoint['epoch']}")
        if 'best_accuracy' in checkpoint:
            print(f"  Best accuracy: {checkpoint['best_accuracy']:.4f}")
    
    def load_data(self, data_path, ref_data_path=None):
        """加载数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load training references from ref_data_path (or fall back to data_path)
        ref_path = ref_data_path if ref_data_path is not None else data_path
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference data file not found: {ref_path}")

        ref_data = np.load(ref_path)

        # Domain A (健康样本)
        self.domain_A_train_X = ref_data['domain_A_train_X']
        self.domain_A_train_Y = ref_data['domain_A_train_Y']

        labels = list(range(9))
        if self.excluded_fault_label is not None:
            labels = [i for i in labels if i != self.excluded_fault_label]

        self.domain_B_train_X_by_class = {}
        self.domain_B_train_Y_by_class = {}

        for i in labels:
            self.domain_B_train_X_by_class[i] = ref_data[f'domain_B_train_X_{i}'][:5]
            self.domain_B_train_Y_by_class[i] = ref_data[f'domain_B_train_Y_{i}'][:5]

        # 测试数据
        # self.test_X = data['test_X']
        # self.test_Y = data['test_Y']
        # Test dataset management
        self.test_sets = {}          # dictionary of named test datasets
        self.active_test_set = None  # currently selected dataset
        self.test_X = None
        self.test_Y = None

        print(f"  Domain A (healthy): {self.domain_A_train_X.shape}")
        print("  Domain B (fault per class):",
            {k: v.shape for k, v in self.domain_B_train_X_by_class.items()})
        print("  Test set: assigned dynamically by dashboard")

    def get_random_fault_references_by_label(self, label, n_samples=2):

        label = int(label)

        if label not in self.domain_B_train_X_by_class:
            raise ValueError(f"Label {label} not available")

        X_class = self.domain_B_train_X_by_class[label]

        if len(X_class) < n_samples:
            raise ValueError("Not enough training samples")

        idx = np.random.choice(len(X_class), n_samples, replace=False)

        return X_class[idx].astype(np.float32)