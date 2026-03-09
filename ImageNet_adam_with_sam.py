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
from common.switcher import DynamicSwitcher
from common.train_flow import train_one_epoch, evaluate
from common.config import get_config, print_config
from common.custom_data_loader import get_data_loaders

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
            img_size=config['image_size']
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


def run_experiment(config):
    strategy_name = config['strategy_name']
    device = config['device']
    print(f"\n===== Training Strategy: {strategy_name} =====")
    
    train_loader, val_loader, test_loader = get_data_loaders(config)
    model = get_model(config)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if strategy_name == "SAM":
        optimizer = SAM(model.parameters(), optim.SGD, rho=config['rho'], lr=config['sam_lr'], momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=config['epochs'], eta_min=1e-6)
    else: # AdamW_Only, AdamW_then_SAM
        optimizer = optim.AdamW(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, total_iters=config['warmup_epochs'])
        main_scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'] - config['warmup_epochs'], eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config['warmup_epochs']])

    switcher = DynamicSwitcher(
        beta_ema=config['beta_ema'],
        history_window=config['history_window'],
        plateau_patience=config['plateau_patience'],
        plateau_min_delta=config['plateau_min_delta'],
        gap_increase_threshold=config['gap_threshold'],
        grad_norm_increase_threshold=config['grad_norm_threshold'],
        min_switch_epoch=config['min_switch_epoch'],
        oscillation_threshold=config['oscillation_threshold']
    )
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'generalization_gap': [], 'test_acc': 0, 'total_training_time': 0}
    best_val_acc = 0.0
    best_model_state = deepcopy(model.state_dict())
    total_training_time = 0
    switched = False

    use_amp = config['use_amp'] 
    print(f"AMP Enabled: {use_amp}")
    
    for epoch in range(config['epochs']):
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
        
        print(f"Epoch {epoch+1:03d}/{config['epochs']} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {end_time:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Gap: {generalization_gap:.2f}%")
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        history['generalization_gap'].append(generalization_gap)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            print(f"----> Best Val Acc Updated: {best_val_acc:.2f}% at epoch {epoch+1}")

        if "SAM" in strategy_name and not switched:
            if switcher.step(epoch=epoch, train_acc=train_acc/100.0, val_acc=val_acc/100.0, grad_norm=avg_grad_norm):
                print(f"\n----- Dynamic Switch Triggered at Epoch {epoch+1}! Switching to SAM -----")
                if strategy_name == "AdamW_then_SAM":
                    optimizer = SAM(model.parameters(), optim.SGD, rho=config['rho'], lr=config['sam_lr'], momentum=0.9)
                elif strategy_name == "AdamW_then_ASAM":
                    optimizer = SAM(model.parameters(), optim.SGD, rho=config['rho'], adaptive=True, lr=config['sam_lr'], momentum=0.9)
                elif strategy_name == "AdamW_then_ESAM":
                    optimizer = ESAM(model.parameters(), optim.SGD, rho=config['rho'], lr=config['esam_lr'], momentum=0.9, beta=0.5)
                
                switched = True
                remaining_epochs = config['epochs'] - epoch
                warmup_scheduler_sam = LinearLR(optimizer.base_optimizer, start_factor=1e-10, end_factor=1.0, total_iters=config['sam_warmup_epochs'])
                main_scheduler_sam = CosineAnnealingLR(optimizer.base_optimizer, T_max=remaining_epochs - config['sam_warmup_epochs'])
                scheduler = SequentialLR(optimizer.base_optimizer, schedulers=[warmup_scheduler_sam, main_scheduler_sam], milestones=[config['sam_warmup_epochs']])
        
        scheduler.step()
        

    model.load_state_dict(best_model_state)
    _, test_acc = evaluate(model, test_loader, criterion, device)
    history['test_acc'] = test_acc
    history['total_training_time'] = total_training_time
    print(f"\n===== Final Test Accuracy for {strategy_name} =====")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Total Training Time: {total_training_time:.2f} sec\n\n")
    
    return history

def plot_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    ax1.set_title("Loss Curves")
    for name, history in results.items():
        ax1.plot(history['train_loss'], '--', label=f'{name} Train') 
        ax1.plot(history['val_loss'], '-', label=f'{name} Validation') 
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Accuracy Curves")
    for name, history in results.items():
        ax2.plot(history['train_acc'], '--', label=f'{name} Train') 
        ax2.plot(history['val_acc'], '-', label=f'{name} Validation') 
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    ax3.set_title("Generalization Gap Curves")
    for name, history in results.items():
        ax3.plot(history['generalization_gap'], '-', label=f'{name}') 
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
        print(f"Final Test Accuracy: {history['test_acc']:.2f}%")
        print(f"Total Training Time: {history['total_training_time']:.2f} sec")
        
def main():
    base_config = get_config()
    print_config(base_config)
    
    strategies_to_run = ["AdamW_then_SAM", "AdamW_Only"] # "AdamW_then_ASAM", "AdamW_then_ESAM",
    all_results = {}
    
    for name in strategies_to_run:
        config = base_config.copy()
        config['strategy_name'] = name
        
        history = run_experiment(config)
        all_results[name] = history
        
    plot_results(all_results)
    print_results(all_results)
    

if __name__ == '__main__':
    set_seed(get_config()['seed'])
    torch.backends.cudnn.benchmark = True 
    main()