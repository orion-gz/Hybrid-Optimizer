import torch

# print config setting
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
        # default experiment 
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_workers': 16,
        
        # dataset 
        'dataset': 'CIFAR100',
        'num_classes': 100,
        'data_path': './data/',
        'image_size': 32,
        
        # data augmentation
        'use_randaugment': False, 
        'use_autoaugment': True, 
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,

        # model
        'model_name': 'vit_small_patch8_224', 
        'dropout_rate': 0.0,
        
        # training
        'use_amp': False, 
        'epochs': EPOCHS,
        'batch_size': 512,
        
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.0001,
        'weight_decay': 0.0005,

        # SAM/ESAM 
        'sam_warmup_epochs': 10,
        'sam_only_lr': 1e-3,  
        'sam_only_rho': 0.20,
        'sam_lr': 0.1,         
        'esam_lr': 0.01,
        'rho': 0.2,
        
        # DynamicSwitcher 
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


def get_config_ver02():
    EPOCHS = 300
    WARMUP_EPOCHS = 10
    
    config = {
        # default experiment
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_workers': 16,
        
        # dataset
        'dataset': 'CIFAR100',
        'num_classes': 100,
        'data_path': './data/',
        'image_size': 32,
        
        # data augmentation
        'use_randaugment': False,
        'use_autoaugment': True,
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,
 
        # model
        'model_name': 'WRN_28_10',
        'dropout_rate': 0.3,
        
        # training
        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 256,
        
        # AdamW 
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.0001,
        'weight_decay': 0.0005,
 
        # SAM
        'sam_only_lr': 1e-3,    # SAM_Only 
        'sam_only_rho': 0.20,   # SAM_Only 
        'rho': 0.05,            
 
        # DynamicSwitcher 
        'min_switch_epoch': 150,            
        'loss_stable_window': 10,           
        'loss_stable_std_threshold': 0.06,  
        
        'slope_window': 30,                 
        'slope_threshold': 0.01,            
        
        'beta_ema': 0.9,
        'history_window': 20,
        'plateau_patience': 20,            
        'plateau_min_delta': 0.005,        
        'gap_threshold': 0.02,
        'grad_norm_threshold': 0.1,
        'oscillation_threshold': 5,
        
        # score weights & threshold
        'w_slope': 0.40,                    # slope signal weights
        'w_plateau': 0.25,                  # plateau signal weights
        'w_gap': 0.20,                      # gap signal weights
        'w_grad': 0.15,                     # grad norm signal weights
        'score_threshold': 0.5,             # switch threshold
    }
    return config

def get_config_ver03():
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
        'model_name': 'deit_tiny_patch16_224',
        'dropout_rate': 0.0,
        
        # 훈련 설정
        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 128,
        
        # AdamW 설정
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,
 
        # SAM 설정
        # ※ sam_lr, sam_warmup_epochs, esam_lr 제거:
        #   전환 시 LR은 AdamW 현재값을 그대로 인계받으므로 별도 설정 불필요
        'sam_only_lr': 1e-3,    # SAM_Only 전략 전용
        'sam_only_rho': 0.20,   # SAM_Only 전략 전용
        'rho': 0.05,            # [변경] 0.2 → 0.05: 수렴 단계 가중치에 맞게 perturbation 축소
 
        # DynamicSwitcher 설정 (파라미터 3개)
        'min_switch_epoch': 150,   # 전환 시도 최소 epoch (전체의 40~60% 권장)
        'slope_window': 30,        # val_acc slope 측정 구간 (20~40 권장)
        'slope_threshold': 0.01,   # %/epoch 이하면 전환 (0.005~0.02 권장)
    }
    return config


def get_config_ver04():
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
        'model_name': 'WRN_28_10',
        'dropout_rate': 0.3,
        
        # 훈련 설정
        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 256,
        
        # AdamW 설정
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,
 
        # SAM 설정
        # ※ sam_lr, sam_warmup_epochs, esam_lr 제거:
        #   전환 시 LR은 AdamW 현재값을 그대로 인계받으므로 별도 설정 불필요
        'sam_only_lr': 1e-3,    # SAM_Only 전략 전용
        'sam_only_rho': 0.20,   # SAM_Only 전략 전용
        'rho': 0.20,            # [변경] 0.2 → 0.05: 수렴 단계 가중치에 맞게 perturbation 축소
 
        # DynamicSwitcher 설정
        'min_switch_epoch': 150,     # 시뮬레이션 시작 최소 epoch (전체의 40~60% 권장)
        'check_every': 10,           # 몇 epoch마다 시뮬레이션할지
        'probe_ratio': 0.1,          # val set 중 시뮬레이션에 사용할 비율
        'sim_steps': 10,             # 시뮬레이션 step 수 (loss spike 구간보다 길게)
        'gain_threshold': 0.2,       # SAM 예측 이득(%) 이 값 이상이면 전환
        'lr_restart_factor': 0.5,    # 전환 시 LR = initial_lr * factor (권장: 0.1~0.5)
    }
    return config

def get_config_ver05():
    """
    ver04 대비 변경점:
      1. rho → switch_rho (0.05): 전환 후 SAM 전용 rho 분리
         - 이미 수렴한 가중치에 rho=0.2는 과도한 perturbation → 급격한 성능 하락
         - switch_rho=0.05로 축소하여 안정적 전환 유도
      2. lr_restart_factor: 0.5 → 0.3
         - v04 로그에서 restart_lr=0.0005는 current_lr 대비 ~10배 점프
         - 0.3으로 낮춰 전환 충격 완화 (restart_lr=0.0003)
      3. DynamicSwitcher_ver05 사용 (공정한 시뮬레이션 비교)
         - SAM과 AdamW 모두 restart_lr로 시뮬레이션
         - "SAM perturbation 자체의 이득"을 순수 측정
    """
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
        'image_size': 32,
 
        # 데이터 증강 설정
        'use_randaugment': False,
        'use_autoaugment': True,
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,
 
        # 모델 설정
        'model_name': 'WRN_28_10',
        'dropout_rate': 0.3,
 
        # 훈련 설정
        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 256,
 
        # AdamW 설정
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,
 
        # SAM 설정
        'sam_only_lr': 1e-3,       # SAM_Only 전략 전용
        'sam_only_rho': 0.20,      # SAM_Only 전략 전용
        # [ver05 핵심 변경] rho를 switch_rho로 분리
        'switch_rho': 0.05,        # 전환 후 SAM에서 사용 (수렴 가중치에 적합)
 
        # DynamicSwitcher_ver05 설정
        'min_switch_epoch': 150,
        'check_every': 10,
        'probe_ratio': 0.1,
        'sim_steps': 10,
        'gain_threshold': 0.2,
        # [ver05 변경] 0.5 → 0.3: 전환 충격 완화
        'lr_restart_factor': 0.3,
    }
    return config

def get_config_ver06():
    """
    ver05 대비 변경점:
 
    [1] rho warmup schedule 도입
        - switch_rho 단일값 → rho_min / rho_max / rho_warmup_epochs
        - 전환 직후 rho_min(0.02)부터 시작, warmup 동안 rho_max(0.15)까지 증가
        - v4의 급락(rho=0.2 즉시) + v5의 무효과(rho=0.05 고정) 동시 해결
 
    [2] 시뮬레이션 신뢰성 강화
        - probe_ratio: 0.1 → 0.3 (측정 분산 감소)
        - sim_steps: 10 → 20 (loss spike 구간 충분히 넘김)
        - gain_threshold: 0.2 → 1.0 (노이즈에 의한 오전환 방지)
 
    [3] rho_max=0.15 선택 근거
        - SAM_Only rho=0.20에서 안정적 학습 확인 (기존 실험)
        - 전환 후에는 이미 수렴한 가중치이므로 약간 보수적인 0.15
        - warmup으로 점진적 도달하므로 0.20보다 안전
    """
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
        'image_size': 32,
 
        # 데이터 증강 설정
        'use_randaugment': False,
        'use_autoaugment': True,
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,
 
        # 모델 설정
        'model_name': 'WRN_28_10',
        'dropout_rate': 0.3,
 
        # 훈련 설정
        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 256,
 
        # AdamW 설정
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,
 
        # SAM 설정
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,
 
        # [ver06 핵심] rho warmup schedule
        'rho_min': 0.02,               # 전환 직후 시작 rho (충격 최소화)
        'rho_max': 0.15,               # warmup 후 최종 rho (flat minima 탐색)
        'rho_warmup_epochs': 20,       # rho가 min→max로 증가하는 구간
 
        # DynamicSwitcher_ver06 설정
        'min_switch_epoch': 150,
        'check_every': 10,
        # [ver06 시뮬레이션 강화]
        'probe_ratio': 0.3,            # 0.1 → 0.3
        'sim_steps': 20,               # 10 → 20
        'gain_threshold': 1.0,         # 0.2 → 1.0
        'lr_restart_factor': 0.3,
    }
    return config

def get_config_ver07():
    """
    ver07: Sharpness 기반 전환 + rho warmup.
 
    ver04~06 네스테로프 시뮬레이션에서의 교훈:
        - SAM은 단기 accuracy에서 AdamW를 이길 수 없음 (원리적 한계)
        - "SAM이 나은가?"가 아니라 "landscape가 sharp한가?"를 측정해야 함
        - rho=0.05는 너무 작아 SAM 효과 없음, rho=0.2는 급락 유발
          → rho warmup (0.02 → 0.15)으로 해결
 
    첫 실험 전략:
        sharpness_threshold=999.0으로 설정하여 전환 없이 sharpness만 기록.
        로그에서 sharpness 추이를 관찰한 뒤 적절한 threshold 결정.
        (AdamW_Only와 동일한 결과가 나오되, sharpness 데이터를 수집)
    """
    EPOCHS = 300
    WARMUP_EPOCHS = 10
 
    config = {
        # 기본 설정
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_workers': 16,
 
        # 데이터셋
        'dataset': 'CIFAR100',
        'num_classes': 100,
        'data_path': './data/',
        'image_size': 32,
 
        # 증강
        'use_randaugment': False,
        'use_autoaugment': True,
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,
 
        # 모델
        'model_name': 'WRN_28_10',
        'dropout_rate': 0.3,
 
        # 훈련
        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 256,
 
        # AdamW
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,
 
        # SAM_Only 전용
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,
 
        # rho warmup (전환 후 SAM에 적용)
        'rho_min': 0.02,
        'rho_max': 0.15,
        'rho_warmup_epochs': 20,
 
        # SharpnessAwareSwitcher
        'min_switch_epoch': 150,
        'check_every': 5,              # 측정이 가벼우므로 자주 가능
        'probe_ratio': 0.2,
        'sharpness_rho': 0.15,         # rho_max와 동일
        'sharpness_threshold': 0.53,  # 첫 실험: 전환 없이 sharpness만 기록
        'sharpness_ema_beta': 0.9,
 
        # LR restart (전환 시)
        'lr_restart_factor': 0.3,
    }
    return config