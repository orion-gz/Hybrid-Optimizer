import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import time
import numpy as np

import matplotlib.pyplot as plt
import random
from copy import deepcopy

from collections import deque
from common.model import WRN_28_10
from common.optimizer import SAM, ESAM
from common.switcher import DynamicSwitcher
from common.train_flow import train_one_epoch_standard, train_one_epoch_sam, train_one_epoch_esam, evaluate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")
print(f"Detected GPU: {torch.cuda.get_device_name(0)}")

# Model Hyperparameter
EPOCHS = 150
WARMUP_EPOCHS = 5
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4

# SAM Hyperparameter
SAM_WARMUP_EPOCHS = 5
SAM_LR = 1e-3
ESAM_LR = 1e-1
RHO = 0.20

# Dynamic Switcher Hyperparmeter
BETA_EMA = 0.90
HISTORY_WINDOW = 15
PLATEAU_PATIENCE = 5
PLATEAU_MIN_DELTA = 0.005
GAP_THRESHOLD = 0.015
GRAD_NORM_THRESHOLD = 0.10
OSCILLATION_THRESHOLD = 5
MIN_SWITCH_EPOCH = int(EPOCHS * 0.1)
FINE_TUNE_EPOCHS = int(EPOCHS * 0.2)

DROPOUT_RATE = 0.3
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
# model = torchvision.models.resnet18(weights=None)
model = WRN_28_10(num_classes=10, dropout_rate=DROPOUT_RATE)

# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 10)
model = model.to(DEVICE)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=WARMUP_EPOCHS)
main_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])

switcher = DynamicSwitcher(
    beta_ema=BETA_EMA,
    history_window=HISTORY_WINDOW,
    plateau_patience=PLATEAU_PATIENCE,
    plateau_min_delta=PLATEAU_MIN_DELTA,
    gap_increase_threshold=GAP_THRESHOLD,
    grad_norm_increase_threshold=GRAD_NORM_THRESHOLD,
    min_switch_epoch=MIN_SWITCH_EPOCH,
    oscillation_threshold=OSCILLATION_THRESHOLD
)

initial_model_state = deepcopy(model.state_dict())
results = {}

strategies = ["SAM", "AdamW_then_SAM", "AdamW_Only"] #"AdamW_then_SAM" "AdamW_then_ESAM", "AdamW_Only"]

for name in strategies:
    print(f"\n===== Training Strategy: {name} =====")
    print(f"-----Hyperparameters------")
    print(f"EPOCHS: {EPOCHS}")
    print(f"WARMUP: {WARMUP_EPOCHS}")
    print(f"LR: {INITIAL_LR}")
    print(f"WEIGHT DECAY: {WEIGHT_DECAY}")
    if "SAM" in name:
        print(f"\n-----SAM Hyperparameters------")    
        print(f"SAM WARMUP: {SAM_WARMUP_EPOCHS}")
        print(f"SAM LR: {SAM_LR}")
        print(f"RHO: {RHO}")
    
        print(f"\n-----Switcher Hyperparameters-----")
        print(switcher)
    print("---------------------------")
    
    model.load_state_dict(initial_model_state)

    if name == "SAM":
        optimizer = SAM(model.parameters(), optim.SGD, rho=RHO, lr=SAM_LR, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=EPOCHS, eta_min=1e-6)
    else: # "AdamW_Only", "AdamW_then_SAM"
        optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, total_iters=WARMUP_EPOCHS)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'generalization_gap': []}
    best_val_acc = 0
    best_model_state = deepcopy(initial_model_state)
    
    switched = False
    patience_counter = 0
    best_val_loss_for_switch = float('inf')
    total_training_time = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        if isinstance(optimizer, SAM):
            train_loss, train_acc, avg_grad_norm = train_one_epoch_sam(model, train_loader, optimizer, criterion, DEVICE)
        elif isinstance(optimizer, ESAM): 
             train_loss, train_acc, avg_grad_norm = train_one_epoch_esam(model, train_loader, optimizer, criterion, DEVICE, gamma=0.5)
        else: 
            train_loss, train_acc, avg_grad_norm = train_one_epoch_standard(model, train_loader, optimizer, criterion, DEVICE)

        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        generalization_gap = train_acc - val_acc
        end_time = time.time() - start_time
        total_training_time += end_time
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {end_time:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Gap: {generalization_gap:.2f}%")
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        history['generalization_gap'].append(generalization_gap)

        if train_loss > 10.0:
            print(f"Epoch {epoch+1}: Loss has diverged! Stopping this training run.")
            history['val_acc'].append(0)
            break 
    
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            print(f"----> Best Val Acc Updated: {best_val_acc:.2f}% at epoch {epoch+1}")
        
        if name != "AdamW_Only" and name != "SAM" and not switched:
            should_switch = switcher.step(
                epoch=epoch,
                train_acc=train_acc / 100.0, 
                val_acc=val_acc / 100.0,   
                grad_norm=avg_grad_norm
            )
            
            # if epoch >= 30:
            #     print(f"\n-----Hard Switch Triggered at Epoch {epoch+1}! Switching to SAM -----")
            #     if name == "AdamW_then_SAM":
            #         optimizer = SAM(model.parameters(), optim.SGD, rho=RHO, lr=SAM_LR, momentum=0.9)
            #     elif name == "AdamW_then_ESAM":
            #         optimizer = ESAM(model.parameters(), optim.SGD, rho=RHO, lr=ESAM_LR, momentum=0.9, beta=0.5)
            #     switched = True
            #     remaining_epochs = EPOCHS - epoch
            #     warmup_scheduler_sam = LinearLR(optimizer.base_optimizer, start_factor=1e-10, end_factor=1.0, total_iters=SAM_WARMUP_EPOCHS)
            #     main_scheduler_sam = CosineAnnealingLR(optimizer.base_optimizer, T_max=remaining_epochs - SAM_WARMUP_EPOCHS)
            #     scheduler = SequentialLR(optimizer.base_optimizer, schedulers=[warmup_scheduler_sam, main_scheduler_sam], milestones=[SAM_WARMUP_EPOCHS])
                
            if should_switch:
                print(f"\n----- Dynamic Switch Triggered at Epoch {epoch+1}! Switching to SAM -----")
                if name == "AdamW_then_SAM":
                    optimizer = SAM(model.parameters(), optim.SGD, rho=RHO, lr=scheduler.get_last_lr()[0], momentum=0.9)
                elif name == "AdamW_then_ESAM":
                    optimizer = ESAM(model.parameters(), optim.SGD, rho=RHO, lr=ESAM_LR, momentum=0.9, beta=0.5)
                switched = True
                remaining_epochs = EPOCHS - epoch
                warmup_scheduler_sam = LinearLR(optimizer.base_optimizer, start_factor=1e-10, end_factor=1.0, total_iters=SAM_WARMUP_EPOCHS)
                main_scheduler_sam = CosineAnnealingLR(optimizer.base_optimizer, T_max=remaining_epochs - SAM_WARMUP_EPOCHS)
                scheduler = SequentialLR(optimizer.base_optimizer, schedulers=[warmup_scheduler_sam, main_scheduler_sam], milestones=[SAM_WARMUP_EPOCHS])
                
            # Cosine Annealing Scheduler 값 기반
            # current_lr = scheduler.get_last_lr()[0]
            
            # if epoch >= WARMUP_EPOCHS and current_lr <= INITIAL_LR * 0.2:
            #     print(f"\nSwitching to {name.split('_then_')[-1]} optimizer at epoch {epoch+1}\n")
            #     if name == "AdamW_then_SAM":
            #         optimizer = SAM(model.parameters(), optim.SGD, rho=RHO, lr=SAM_LR, momentum=0.9)
            #     elif name == "AdamW_then_ESAM":
            #         optimizer = ESAM(model.parameters(), optim.SGD, rho=RHO, lr=ESAM_LR, momentum=0.9, beta=0.5)
            #     switched = True
            #     remaining_epochs = EPOCHS - epoch
            #     scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=remaining_epochs)
            
        if scheduler != None:
            scheduler.step()

    # if name != "AdamW_Only" and not switched:
    #     print(f"\n===== Switch did not trigger. Starting Post-Hoc SAM Fine-tuning for {FINE_TUNE_EPOCHS} epochs. =====")
    #     model.load_state_dict(best_model_state)
    #     FINE_TUNE_SAM_LR = 0.1
    #     base_optimizer = optim.SGD
    #     optimizer = SAM(model.parameters(), base_optimizer, rho=RHO, lr=FINE_TUNE_SAM_LR, momentum=0.9)
    #     scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=FINE_TUNE_EPOCHS, eta_min=1e-6)

    #     for ft_epoch in range(FINE_TUNE_EPOCHS):
    #         epoch = EPOCHS + ft_epoch 

    #         train_loss, train_acc, _ = train_one_epoch_sam(model, train_loader, optimizer, criterion, DEVICE)
    #         val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
            
    #         print(f"Epoch {epoch+1:03d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {end_time:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Gap: {generalization_gap:.2f}%")
    #         if val_acc >= best_val_acc:
    #             best_val_acc = val_acc
    #             best_model_state = deepcopy(model.state_dict())
    #             print(f"----> Best Val Acc Updated during Fine-tuning: {best_val_acc:.2f}%")

    #         scheduler.step()

    print(f"\n===== Final Evaluation for {name} =====")
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Final Test Accuracy: {test_acc:.2f}%\n")
    print(f"Total Training Time: {total_training_time:.2f} sec\n")
    results[name] = history
    results[name]['test_acc'] = test_acc

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