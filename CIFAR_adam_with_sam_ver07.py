import os
from models.CompactTransformers.src import cct_7_3x1_32_c100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment

import timm

from common.model import WRN_28_10
from common.optimizer import SAM, ESAM
from common.switcher import SharpnessAwareSwitcher
from common.train_flow import train_one_epoch, evaluate
from common.config import get_config_ver07, print_config

import matplotlib.pyplot as plt
import numpy as np
import time
import random
from copy import deepcopy

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_data_loaders(config):
    DATASET_STATS = {
        'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]},
        'CIFAR100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
        'SVHN': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    }
    DATASET_CLASSES = {
        'CIFAR10': torchvision.datasets.CIFAR10,
        'CIFAR100': torchvision.datasets.CIFAR100,
        'SVHN': torchvision.datasets.SVHN,
    }
    
    dataset_name = config['dataset']
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    mean = DATASET_STATS[dataset_name]['mean']
    std = DATASET_STATS[dataset_name]['std']
    dataset_class = DATASET_CLASSES[dataset_name]

    train_transforms_list = []
    test_transforms_list = []
    
    if "patch16" in config['model_name']:
        train_transforms_list.append(transforms.Resize((224, 224), antialias=True))
        train_transforms_list.append(transforms.RandomCrop(224, padding=32))
        train_transforms_list.append(transforms.RandomHorizontalFlip())
        test_transforms_list.append(transforms.Resize((224, 224), antialias=True))
    else:
        train_transforms_list.append(transforms.RandomCrop(32, padding=4))
        train_transforms_list.append(transforms.RandomHorizontalFlip())
        
    if config.get('use_autoaugment', False):
        if dataset_name == 'SVHN':
            train_transforms_list.append(AutoAugment(policy=AutoAugmentPolicy.SVHN))
        elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
            train_transforms_list.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))
            
    if config.get('use_randaugment', False):
        train_transforms_list.append(RandAugment(num_ops=2, magnitude=9))
    
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    train_transform = transforms.Compose(train_transforms_list)
    test_transform = transforms.Compose(test_transforms_list)
    
    if dataset_name == 'SVHN':
        full_train_dataset = dataset_class(root=config['data_path'], split='train', download=True, transform=train_transform)
        test_dataset = dataset_class(root=config['data_path'], split='test', download=True, transform=test_transform)
    else:
        full_train_dataset = dataset_class(root=config['data_path'], train=True, download=True, transform=train_transform)
        test_dataset = dataset_class(root=config['data_path'], train=False, download=True, transform=test_transform)
        
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    return train_loader, val_loader, test_loader

def get_model(config):
    if config['model_name'] == 'WRN_28_10':
        model = WRN_28_10(num_classes=config['num_classes'], dropout_rate=config['dropout_rate'])
    elif config['model_name'] == 'resnet18':
        model = models.resnet18(weights=None, num_classes=config['num_classes'])
    elif config['model_name'] == 'resnet50':
        model = models.resnet50(weights=None, num_classes=config['num_classes'])
    elif config['model_name'] == 'cct_7_3x1_32_c100':
        model = cct_7_3x1_32_c100()
    elif config['model_name'] == 'vit_small_patch8_224':
        model = timm.create_model(
            'vit_small_patch8_224', 
            pretrained=True,
            num_classes=config['num_classes'],
            img_size=32 
        )
    elif 'patch16' in config['model_name']:
        model = timm.create_model(
            config['model_name'],
            pretrained=True,      
            num_classes=config['num_classes'], 
            drop_rate=config['dropout_rate'], 
            drop_path_rate=0.1      
        )
    elif config['model_name'] == 'efficientnetv2_s':
        model = timm.create_model('efficientnetv2_s', pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")
    
    return model.to(config['device'])


# ═══════════════════════════════════════════════════════════════════
#  전환 관련 함수
# ═══════════════════════════════════════════════════════════════════

def switch_to_sam(model, optimizer, config):
    """
    AdamW -> SAM(AdamW base) 전환.
    
    rho warmup 전략:
        rho_min(0.02)부터 시작 -> rho_warmup_epochs(20) 동안 rho_max(0.15)까지 증가.
        전환 초기 급락 방지 + 최종적으로 충분한 perturbation 확보.
    """
    current_lr = optimizer.param_groups[0]['lr']
    restart_lr = config['initial_lr'] * config['lr_restart_factor']
    rho_min = config['rho_min']
    rho_max = config['rho_max']

    print(f"    └─ AdamW current LR : {current_lr:.6f}")
    print(f"    └─ SAM restart LR   : {restart_lr:.6f}"
          f"  (initial_lr={config['initial_lr']} x factor={config['lr_restart_factor']})")
    print(f"    └─ rho warmup: {rho_min} -> {rho_max} over {config['rho_warmup_epochs']} epochs")
    print(f"    └─ base_optimizer: AdamW")

    new_optimizer = SAM(
        model.parameters(),
        optim.AdamW,
        rho=rho_min,         # rho_min으로 시작
        lr=restart_lr,
        #momentum=0.9
        weight_decay=config['weight_decay']
    )

    # AdamW momentum state 이전
    adamw_state = optimizer.state_dict()['state']
    param_list = list(model.parameters())
    for i, param in enumerate(param_list):
        if i in adamw_state:
            new_optimizer.base_optimizer.state[param] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in adamw_state[i].items()
            }

    return new_optimizer, restart_lr


def update_rho(optimizer, epoch, switch_epoch, config):
    """
    rho warmup: 전환 후 rho_min -> rho_max 선형 증가.
    warmup 완료 후에는 rho_max 고정.
    """
    elapsed = epoch - switch_epoch
    warmup = config['rho_warmup_epochs']
    rho_min = config['rho_min']
    rho_max = config['rho_max']

    progress = min(1.0, elapsed / max(warmup, 1))
    current_rho = rho_min + (rho_max - rho_min) * progress

    for group in optimizer.param_groups:
        group['rho'] = current_rho

    return current_rho


# ═══════════════════════════════════════════════════════════════════
#  실험 실행
# ═══════════════════════════════════════════════════════════════════

def run_experiment(config):
    strategy_name = config['strategy_name']
    device = config['device']
    print(f"\n===== Training Strategy: {strategy_name} =====")
    
    train_loader, val_loader, test_loader = get_data_loaders(config)
    model = get_model(config)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # ── optimizer & scheduler 초기화 ────────────────────────────
    if strategy_name == "SAM_Only":
        optimizer = SAM(
            model.parameters(),
            optim.AdamW,
            rho=config['sam_only_rho'],
            lr=config['initial_lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = CosineAnnealingLR(
            optimizer.base_optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
    else:  # AdamW_Only, AdamW_then_*
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['initial_lr'],
            weight_decay=config['weight_decay']
        )
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, total_iters=config['warmup_epochs'])
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        #CosineAnnealingLR(
        #    optimizer,
        #    T_max=config['epochs'] - config['warmup_epochs'],
        #    eta_min=1e-6
        #)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config['warmup_epochs']]
        )

    # ── switcher 초기화 ─────────────────────────────────────────
    switcher = SharpnessAwareSwitcher(
        min_switch_epoch=config['min_switch_epoch'],
        check_every=config['check_every'],
        probe_ratio=config['probe_ratio'],
        sharpness_rho=config['sharpness_rho'],
        sharpness_threshold=config['sharpness_threshold'],
        sharpness_ema_beta=config['sharpness_ema_beta'],
    )
    
    # ── 학습 상태 초기화 ────────────────────────────────────────
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'generalization_gap': [],
        'switch_epoch': None,
        'test_acc': 0,
        'total_training_time': 0
    }
    best_val_acc = 0.0
    best_model_state = deepcopy(model.state_dict())
    total_training_time = 0
    switched = False
    switch_epoch_num = None   # rho warmup 기준점
    
    use_amp = config['use_amp']
    print(f"AMP Enabled: {use_amp}")
    
    # ── 학습 루프 ───────────────────────────────────────────────
    for epoch in range(config['epochs']):
        
        # rho warmup: SAM 전환 후 매 epoch rho 업데이트
        current_rho = None
        if switched and switch_epoch_num is not None:
            current_rho = update_rho(optimizer, epoch, switch_epoch_num, config)

        start_time = time.time()
        
        train_loss, train_acc, avg_grad_norm = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=config['use_mixup'], use_cutmix=config['use_cutmix'],
            mixup_alpha=config.get('mixup_alpha', 1.0),
            use_amp=use_amp
        )
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        generalization_gap = train_acc - val_acc
        end_time = time.time() - start_time
        total_training_time += end_time
        
        # 로그 출력
        phase = "[SAM]" if switched else "[AdamW]"
        rho_str = f" | rho: {current_rho:.4f}" if current_rho is not None else ""
        print(
            f"Epoch {epoch+1:03d}/{config['epochs']} {phase} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}{rho_str} | "
            f"Time: {end_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Gap: {generalization_gap:.2f}%"
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['generalization_gap'].append(generalization_gap)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            print(f"----> Best Val Acc Updated: {best_val_acc:.2f}% at epoch {epoch+1}")

        # ── 전환 로직 (Sharpness 기반) ─────────────────────────
        if "then" in strategy_name and not switched:
            # SharpnessAwareSwitcher: optimizer 불필요 (가중치 상태만 측정)
            if switcher.step(
                epoch=epoch,
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            ):
                print(f"\n----- Sharpness Switch Triggered at Epoch {epoch+1}! -----")
                history['switch_epoch'] = epoch + 1
                switch_epoch_num = epoch + 1

                if strategy_name in ("AdamW_then_SAM", "AdamW_then_ASAM"):
                    adaptive = (strategy_name == "AdamW_then_ASAM")
                    new_optimizer, _ = switch_to_sam(model, optimizer, config)
                    if adaptive:
                        for group in new_optimizer.param_groups:
                            group['adaptive'] = True
                    optimizer = new_optimizer

                elif strategy_name == "AdamW_then_ESAM":
                    restart_lr = config['initial_lr'] * config['lr_restart_factor']
                    print(f"    └─ ESAM restart LR: {restart_lr:.6f}")
                    optimizer = ESAM(
                        model.parameters(),
                        optim.AdamW,
                        rho=config['rho_min'],
                        lr=restart_lr,
                        weight_decay=config['weight_decay'],
                        beta=0.5
                    )

                switched = True
                remaining_epochs = config['epochs'] - (epoch + 1)

                scheduler = CosineAnnealingLR(
                    optimizer.base_optimizer,
                    T_max=max(remaining_epochs, 1),
                    eta_min=1e-6
                )
                print(f"    └─ Cosine scheduler restarted: T_max={remaining_epochs} epochs")
                print(f"----- Switch Complete -----\n")
        # ───────────────────────────────────────────────────────

        scheduler.step()

    # ── 최종 평가 ───────────────────────────────────────────────
    model.load_state_dict(best_model_state)
    _, test_acc = evaluate(model, test_loader, criterion, device)
    history['test_acc'] = test_acc
    history['total_training_time'] = total_training_time
    
    save_dir = "/home/prml/StudentsWork/JungWoo/Optimizer/results/CIFAR100/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_name = config['model_name']
    save_filename = f"{model_name}_{strategy_name}_best.pth"
    save_path = os.path.join(save_dir, save_filename)
    torch.save(best_model_state, save_path)
    print(f"----> Model saved to: {save_path}")
    
    print(f"\n===== Final Test Accuracy for {strategy_name} =====")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    if history['switch_epoch']:
        print(f"Switch Epoch: {history['switch_epoch']}")
    print(f"Total Training Time: {total_training_time:.2f} sec\n\n")
    
    return history


# ═══════════════════════════════════════════════════════════════════
#  시각화 & 출력
# ═══════════════════════════════════════════════════════════════════

def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    ax1, ax2, ax3 = axes

    ax1.set_title("Loss Curves")
    for name, history in results.items():
        ax1.plot(history['train_loss'], '--', label=f'{name} Train')
        ax1.plot(history['val_loss'], '-', label=f'{name} Val')
        if history.get('switch_epoch'):
            ax1.axvline(x=history['switch_epoch'], linestyle=':', alpha=0.7, label=f'{name} Switch')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Accuracy Curves")
    for name, history in results.items():
        ax2.plot(history['train_acc'], '--', label=f'{name} Train')
        ax2.plot(history['val_acc'], '-', label=f'{name} Val')
        if history.get('switch_epoch'):
            ax2.axvline(x=history['switch_epoch'], linestyle=':', alpha=0.7, label=f'{name} Switch')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    ax3.set_title("Generalization Gap Curves")
    for name, history in results.items():
        ax3.plot(history['generalization_gap'], '-', label=f'{name}')
        if history.get('switch_epoch'):
            ax3.axvline(x=history['switch_epoch'], linestyle=':', alpha=0.7, label=f'{name} Switch')
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Generalization Gap (%)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('optimization_results.png')
    plt.show()

def print_results(results):
    print("\n====== All Final Results =====\n")
    for name, history in results.items():
        print(f"===== {name} =====")
        print(f"Final Test Accuracy : {history['test_acc']:.2f}%")
        if history.get('switch_epoch'):
            print(f"Switch Epoch       : {history['switch_epoch']}")
        print(f"Total Training Time : {history['total_training_time']:.2f} sec")
        
def main():
    base_config = get_config_ver07()
    print_config(base_config)
    
    strategies_to_run = ["AdamW_then_SAM"]
    all_results = {}
    
    
    for name in strategies_to_run:
        config = base_config.copy()
        config['strategy_name'] = name
        history = run_experiment(config)
        all_results[name] = history
        
    plot_results(all_results)
    print_results(all_results)

if __name__ == '__main__':
    set_seed(get_config_ver07()['seed'])
    torch.backends.cudnn.benchmark = True
    main()