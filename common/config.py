import torch

def print_config(config):
    max_key_len = max(len(key) for key in config.keys())
    print("=" * (max_key_len + 5))
    for key, value in config.items():
        if key == '\n':
            print()
        else:
            print(f"{key:<{max_key_len}} : {value}")        
    print("=" * (max_key_len + 5))
    
def get_config():
    EPOCHS = 300
    WARMUP_EPOCHS = 10
    
    config = {
        # 기본 설정
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_workers': 16,
        
        # 데이터셋 설정
        'dataset': 'CIFAR100',
        'num_classes': 100,
        'data_path': './data/',
        'image_size': 224,
        
        # 데이터 증강 설정
        'use_randaugment': False, 
        'use_autoaugment': True, 
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,

        # 모델 설정
        # WRN_28_10, resnet18, cct_7_3x1_32_c100, vit_small_patch8_224, vit_tiny_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224, efficientnetv2_s
        'model_name': 'deit_tiny_patch16_224', 
        'dropout_rate': 0.0,
        
        # 훈련 설정
        'use_amp': False, 
        'epochs': EPOCHS,
        'batch_size': 128,
        
        # AdamW 설정 (ViT 초기 학습용)
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,

        # SAM/ESAM 설정 (분리)
        'sam_warmup_epochs': 10,
        'sam_only_lr': 1e-3,  
        'sam_only_rho': 0.20,
        'sam_lr': 0.1,         
        'esam_lr': 0.01,
        'rho': 0.2,
        
        # DynamicSwitcher 설정
        'beta_ema': 0.9,
        'history_window': 20,
        'plateau_patience': 10,
        'plateau_min_delta': 0.01,
        'gap_threshold': 0.02,
        'grad_norm_threshold': 0.1,
        'oscillation_threshold': 5,
        'min_switch_epoch': 30.0,
    }
    return config