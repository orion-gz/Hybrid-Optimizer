"""
CIFAR_adam_with_sam_Forced_Switch.py

고정 비율 전환 실험: AdamW(constant LR) → SAM(cosine decay)
전환 시점을 [50, 75, 100, 150, 200]으로 고정하여
최적 AdamW:SAM 비율을 탐색.

설정:
    - AdamW phase: warmup 10ep + constant LR=0.001 (cosine decay 없음)
    - SAM phase: LR restart 0.0003, cosine decay (T_max=remaining), rho warmup
    - v7_03과 동일한 조건에서 switch epoch만 변경

비교 대상 (이미 확보):
    - AdamW_Only (constant LR): 77.93%
    - v7_03 (sharpness 기반, ep151 전환): 81.24%
    - AdamW_Only (cosine): 81.45%
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment

import timm

from common.model import WRN_28_10
from common.optimizer import SAM
from common.train_flow import train_one_epoch, evaluate

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


def get_config_forced_switch():
    EPOCHS = 300
    WARMUP_EPOCHS = 10

    config = {
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_workers': 16,

        'dataset': 'CIFAR100',
        'num_classes': 100,
        'data_path': './data/',
        'image_size': 32,

        'use_randaugment': False,
        'use_autoaugment': True,
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,

        'model_name': 'WRN_28_10',
        'dropout_rate': 0.3,

        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 256,

        # AdamW 설정 (constant LR)
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,

        # SAM 설정
        'rho_min': 0.02,
        'rho_max': 0.15,
        'rho_warmup_epochs': 20,

        # 전환 설정
        'lr_restart_factor': 0.3,       # restart_lr = 0.001 * 0.3 = 0.0003
        'forced_switch_epoch': 100,     # main()에서 덮어씀
    }
    return config


def print_config(config):
    print("=" * 24)
    for k, v in config.items():
        print(f"{k:<20}: {v}")
    print("=" * 24)


# ═══════════════════════════════════════════════════════════════════
#  데이터 & 모델
# ═══════════════════════════════════════════════════════════════════

def get_data_loaders(config):
    DATASET_STATS = {
        'CIFAR10':  {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]},
        'CIFAR100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
    }
    dataset_name = config['dataset']
    mean = DATASET_STATS[dataset_name]['mean']
    std  = DATASET_STATS[dataset_name]['std']

    dataset_class = {
        'CIFAR10':  torchvision.datasets.CIFAR10,
        'CIFAR100': torchvision.datasets.CIFAR100,
    }[dataset_name]

    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if config.get('use_autoaugment', False):
        train_transforms_list.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))
    if config.get('use_randaugment', False):
        train_transforms_list.append(RandAugment(num_ops=2, magnitude=9))
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    train_transform = transforms.Compose(train_transforms_list)
    test_transform  = transforms.Compose(test_transforms_list)

    full_train = dataset_class(root=config['data_path'], train=True,  download=True, transform=train_transform)
    test_set   = dataset_class(root=config['data_path'], train=False, download=True, transform=test_transform)

    train_size = int(0.8 * len(full_train))
    val_size   = len(full_train) - train_size
    gen = torch.Generator().manual_seed(config['seed'])
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,  num_workers=config['num_workers'])
    val_loader   = DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader  = DataLoader(test_set,  batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader, test_loader


def get_model(config):
    if config['model_name'] == 'WRN_28_10':
        model = WRN_28_10(num_classes=config['num_classes'], dropout_rate=config['dropout_rate'])
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")
    return model.to(config['device'])


# ═══════════════════════════════════════════════════════════════════
#  전환 함수
# ═══════════════════════════════════════════════════════════════════

def switch_to_sam(model, optimizer, config):
    """AdamW → SAM(AdamW base) 전환. rho_min으로 시작, momentum state 이전."""
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
        rho=rho_min,
        lr=restart_lr,
        weight_decay=config['weight_decay']
    )

    # AdamW momentum/variance state 이전
    adamw_state = optimizer.state_dict()['state']
    param_list  = list(model.parameters())
    for i, param in enumerate(param_list):
        if i in adamw_state:
            new_optimizer.base_optimizer.state[param] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in adamw_state[i].items()
            }

    return new_optimizer, restart_lr


def update_rho(optimizer, epoch, switch_epoch, config):
    """rho warmup: rho_min → rho_max 선형 증가."""
    elapsed  = epoch - switch_epoch
    warmup   = config['rho_warmup_epochs']
    rho_min  = config['rho_min']
    rho_max  = config['rho_max']

    progress    = min(1.0, elapsed / max(warmup, 1))
    current_rho = rho_min + (rho_max - rho_min) * progress

    for group in optimizer.param_groups:
        group['rho'] = current_rho

    return current_rho


# ═══════════════════════════════════════════════════════════════════
#  실험 실행
# ═══════════════════════════════════════════════════════════════════

def run_experiment(config):
    strategy_name     = config['strategy_name']
    device            = config['device']
    forced_switch_ep  = config.get('forced_switch_epoch', None)

    label = strategy_name
    if forced_switch_ep is not None and "then" in strategy_name:
        label = f"{strategy_name}_switch@{forced_switch_ep}"

    print(f"\n{'='*60}")
    print(f"  Training Strategy: {label}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = get_data_loaders(config)
    model     = get_model(config)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── optimizer & scheduler 초기화 ────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['initial_lr'],
        weight_decay=config['weight_decay']
    )

    # AdamW phase: warmup 10ep → constant LR (cosine decay 없음)
    warmup_scheduler   = LinearLR(optimizer, start_factor=1e-10, total_iters=config['warmup_epochs'])
    constant_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler],
        milestones=[config['warmup_epochs']]
    )

    # ── 학습 상태 ──────────────────────────────────────────────
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'generalization_gap': [],
        'switch_epoch': None,
        'test_acc': 0,
        'total_training_time': 0,
    }
    best_val_acc      = 0.0
    best_model_state  = deepcopy(model.state_dict())
    total_training_time = 0
    switched          = False
    switch_epoch_num  = None

    use_amp = config['use_amp']
    print(f"AMP Enabled: {use_amp}")
    if forced_switch_ep is not None:
        print(f"Forced Switch Epoch: {forced_switch_ep}")

    # ── 학습 루프 ───────────────────────────────────────────────
    for epoch in range(config['epochs']):

        # rho warmup
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
        elapsed = time.time() - start_time
        total_training_time += elapsed

        # 로그
        phase   = "[SAM]" if switched else "[AdamW]"
        rho_str = f" | rho: {current_rho:.4f}" if current_rho is not None else ""
        print(
            f"Epoch {epoch+1:03d}/{config['epochs']} {phase} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}{rho_str} | "
            f"Time: {elapsed:.2f}s | "
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

        # ── 고정 비율 전환 ─────────────────────────────────────
        if "then" in strategy_name and not switched:
            if forced_switch_ep is not None and (epoch + 1) >= forced_switch_ep:

                print(f"\n----- Forced Switch at Epoch {epoch+1}! -----")
                history['switch_epoch'] = epoch + 1
                switch_epoch_num = epoch + 1

                new_optimizer, _ = switch_to_sam(model, optimizer, config)
                optimizer = new_optimizer
                switched  = True

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
    save_filename = f"{model_name}_{label}_best.pth"
    save_path = os.path.join(save_dir, save_filename)
    torch.save(best_model_state, save_path)
    print(f"----> Model saved to: {save_path}")

    print(f"\n===== Final Test Accuracy for {label} =====")
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
        ax1.plot(history['val_loss'], '-', label=f'{name}')
        if history.get('switch_epoch'):
            ax1.axvline(x=history['switch_epoch'], linestyle=':', alpha=0.5)
    ax1.set_xlabel("Epochs"); ax1.set_ylabel("Val Loss")
    ax1.legend(fontsize=8); ax1.grid(True)

    ax2.set_title("Val Accuracy Curves")
    for name, history in results.items():
        ax2.plot(history['val_acc'], '-', label=f'{name}')
        if history.get('switch_epoch'):
            ax2.axvline(x=history['switch_epoch'], linestyle=':', alpha=0.5)
    ax2.set_xlabel("Epochs"); ax2.set_ylabel("Val Accuracy (%)")
    ax2.legend(fontsize=8); ax2.grid(True)

    ax3.set_title("Switch Epoch vs Test Accuracy")
    switch_eps  = []
    test_accs   = []
    for name, history in results.items():
        if history.get('switch_epoch'):
            switch_eps.append(history['switch_epoch'])
            test_accs.append(history['test_acc'])
    if switch_eps:
        ax3.plot(switch_eps, test_accs, 'o-', markersize=8, linewidth=2)
        for x, y in zip(switch_eps, test_accs):
            ax3.annotate(f'{y:.2f}%', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)
        # baselines
        ax3.axhline(y=77.93, color='gray', linestyle='--', alpha=0.5, label='AdamW_Only(const) 77.93%')
        ax3.axhline(y=81.45, color='blue', linestyle='--', alpha=0.5, label='AdamW_Only(cosine) 81.45%')
        ax3.legend(fontsize=8)
    ax3.set_xlabel("Switch Epoch"); ax3.set_ylabel("Test Accuracy (%)")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('forced_switch_results.png', dpi=150)
    plt.show()


def print_results(results):
    print("\n" + "=" * 70)
    print("  All Final Results — Forced Switch Experiment")
    print("=" * 70 + "\n")

    print(f"{'Experiment':<35} {'Switch Ep':>10} {'Test Acc':>10} {'Best Val':>10} {'Time':>10}")
    print("-" * 78)

    for name, h in results.items():
        se = h.get('switch_epoch', '—')
        ta = f"{h['test_acc']:.2f}%"
        bv = f"{max(h['val_acc']):.2f}%"
        tt = f"{h['total_training_time']:.0f}s"
        print(f"{name:<35} {str(se):>10} {ta:>10} {bv:>10} {tt:>10}")

    # 기존 결과도 함께 출력
    print(f"\n--- Reference (이전 실험) ---")
    print(f"{'AdamW_Only (constant LR)':<35} {'—':>10} {'77.93%':>10} {'72.14%':>10} {'24451s':>10}")
    print(f"{'v7_03 sharpness@151':<35} {'151':>10} {'81.24%':>10} {'76.98%':>10} {'36044s':>10}")
    print(f"{'AdamW_Only (cosine)':<35} {'—':>10} {'81.45%':>10} {'76.89%':>10} {'24526s':>10}")


# ═══════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════

def main():
    base_config = get_config_forced_switch()
    print_config(base_config)

    # ── 고정 비율 전환 실험 ─────────────────────────────────────
    switch_epochs = [50, 75, 100, 150, 200]
    all_results = {}

    for se in switch_epochs:
        # 매 실험마다 seed 재설정 (동일 초기 조건 보장)
        set_seed(base_config['seed'])

        config = base_config.copy()
        config['strategy_name'] = 'AdamW_then_SAM'
        config['forced_switch_epoch'] = se

        history = run_experiment(config)
        all_results[f'switch@{se}'] = history

    # ── 결과 출력 & 시각화 ──────────────────────────────────────
    plot_results(all_results)
    print_results(all_results)


if __name__ == '__main__':
    set_seed(get_config_forced_switch()['seed'])
    torch.backends.cudnn.benchmark = True
    main()