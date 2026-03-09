import random
import numpy as np
import torch
import torch.nn as nn
from common.optimizer import SAM, ESAM
from common.augmentation import rand_bbox, mixup_cutmix_data, mixup_cutmix_criterion
from torch.amp import autocast, GradScaler # AMP 관련 모듈 import

def get_total_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

import math

def sam_step(model, optimizer, criterion, inputs, label_info, mixup_applied, scaler):
    # 1. First Step
    with autocast('cuda', enabled=scaler is not None):
        outputs = model(inputs)
        loss = mixup_cutmix_criterion(criterion, outputs, *label_info) if mixup_applied else criterion(outputs, *label_info)
    
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) 
        
        grad_norm = get_total_grad_norm(model)
        # [핵심 방어 1] 첫 번째 패스에서 NaN/Inf 발생 시 업데이트 스킵
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            optimizer.zero_grad()
            scaler.update() # Scale Factor 감소 유도
            return loss, grad_norm, outputs
            
        optimizer.first_step(zero_grad=True)
    else:
        loss.backward()
        grad_norm = get_total_grad_norm(model)
        optimizer.first_step(zero_grad=True)
    
    # 2. Second Step
    with autocast('cuda', enabled=scaler is not None):
        outputs2 = model(inputs)
        second_loss = mixup_cutmix_criterion(criterion, outputs2, *label_info) if mixup_applied else criterion(outputs2, *label_info)
    
    if scaler:
        scaler.scale(second_loss).backward() # 언더플로우 방지를 위해 다시 스케일링
        
        # [핵심 방어 2] unscale_() 중복 호출 에러를 우회하는 '수동 unscale' 및 Inf 검사
        inv_scale = 1.0 / scaler.get_scale()
        found_inf = False
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(inv_scale)
                if not torch.isfinite(p.grad).all():
                    found_inf = True
        
        if found_inf:
            # 두 번째 패스에서 폭발 시, 가중치 붕괴를 막기 위해 old_p로 안전하게 롤백
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p in optimizer.state and 'old_p' in optimizer.state[p]:
                        p.data = optimizer.state[p]['old_p'].clone()
            optimizer.zero_grad()
        else:
            optimizer.second_step(zero_grad=True)
            
        scaler.update()
    else:
        second_loss.backward()
        optimizer.second_step(zero_grad=True)

    return loss, grad_norm, outputs

def esam_step(model, optimizer, criterion, inputs, label_info, mixup_applied, gamma, scaler):
    criterion_none = nn.CrossEntropyLoss(reduction='none') 
    
    # 1. First Step
    with autocast('cuda', enabled=scaler is not None):
        outputs = model(inputs)
        if mixup_applied:
            targets_a, targets_b, lam = label_info
            base_losses = lam * criterion_none(outputs, targets_a) + (1 - lam) * criterion_none(outputs, targets_b)
        else:
            targets, = label_info
            base_losses = criterion_none(outputs, targets)
        loss = base_losses.mean()

    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        grad_norm = get_total_grad_norm(model)
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            optimizer.zero_grad()
            scaler.update()
            return loss, grad_norm, outputs
            
        optimizer.first_step(zero_grad=True)
    else:
        loss.backward()
        grad_norm = get_total_grad_norm(model)
        optimizer.first_step(zero_grad=True)
    
    # ESAM Selection Logic
    with torch.no_grad():
        with autocast('cuda', enabled=scaler is not None):
            perturbed_outputs = model(inputs)
            if mixup_applied:
                perturbed_losses = lam * criterion_none(perturbed_outputs, targets_a) + (1 - lam) * criterion_none(perturbed_outputs, targets_b)
            else:
                perturbed_losses = criterion_none(perturbed_outputs, targets)
        
    loss_increases = perturbed_losses - base_losses
    _, indices = torch.topk(loss_increases, int(gamma * inputs.size(0)))
    
    # 2. Second Step
    final_inputs = inputs[indices]
    if mixup_applied:
        targets_a_sel, targets_b_sel = targets_a[indices], targets_b[indices]
        
    with autocast('cuda', enabled=scaler is not None):
        final_outputs = model(final_inputs)
        if mixup_applied:
            selected_loss = mixup_cutmix_criterion(criterion, final_outputs, targets_a_sel, targets_b_sel, lam)
        else:
            selected_loss = criterion(final_outputs, targets[indices])
            
    if scaler:
        scaler.scale(selected_loss).backward()
        
        inv_scale = 1.0 / scaler.get_scale()
        found_inf = False
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(inv_scale)
                if not torch.isfinite(p.grad).all():
                    found_inf = True
                    
        if found_inf:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p in optimizer.state and 'old_p' in optimizer.state[p]:
                        p.data = optimizer.state[p]['old_p'].clone()
            optimizer.zero_grad()
        else:
            optimizer.second_step(zero_grad=True)
            
        scaler.update()
    else:
        selected_loss.backward()
        optimizer.second_step(zero_grad=True)
    
    return loss, grad_norm, final_outputs

def step(model, optimizer, criterion, inputs, label_info, mixup_applied, scaler):  
    optimizer.zero_grad()
    
    with autocast('cuda', enabled=scaler is not None):
        outputs = model(inputs)
        loss = mixup_cutmix_criterion(criterion, outputs, *label_info) if mixup_applied else criterion(outputs, *label_info)
    
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    return loss, get_total_grad_norm(model), outputs
    
def update_step(model, optimizer, criterion, inputs, label_info, mixup_applied, gamma, scaler):
    if isinstance(optimizer, SAM):
        return sam_step(model, optimizer, criterion, inputs, label_info, mixup_applied, scaler)
    elif isinstance(optimizer, ESAM):
        return esam_step(model, optimizer, criterion, inputs, label_info, mixup_applied, gamma, scaler)
    else: # Standard (AdamW, SGD etc.)
        return step(model, optimizer, criterion, inputs, label_info, mixup_applied, scaler)
    
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_mixup: bool = False,
    use_cutmix: bool = False,
    mixup_alpha: float = 1.0,
    gamma: float = 0.5,
    use_amp: bool = True # AMP 사용 여부 추가
):
    model.train()
    total_loss, total_grad_norm, correct, total = 0.0, 0.0, 0, 0
    
    # Scaler 초기화 (AMP 사용할 때만)
    scaler = GradScaler() if use_amp else None
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        final_inputs, label_info, mixup_applied = mixup_cutmix_data(inputs, targets, use_mixup, use_cutmix, mixup_alpha)
            
        loss, grad_norm, outputs = update_step(model, optimizer, criterion, final_inputs, label_info, mixup_applied, gamma, scaler)
        
        # ESAM은 subset output을 반환하므로 정확도 계산 시 주의 필요 (여기서는 전체 배치의 loss로 근사하거나 무시)
        # 정확도 계산을 위해 예측값 복원 (Mixup 시에는 argmax가 부정확할 수 있음)
        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        total_grad_norm += grad_norm
        
        # 훈련 중 정확도는 참고용
        if not mixup_applied:
            # ESAM의 경우 outputs 크기가 다를 수 있음 -> 간단히 전체 배치에 대해 다시 forward하지 않고,
            # 속도를 위해 여기서는 정확도 계산을 건너뛰거나, step 함수가 전체 출력을 반환하도록 수정해야 함.
            # 위 구현에서는 SAM/Standard는 전체 outputs, ESAM은 subset outputs을 반환함.
            # 편의상 ESAM일때는 정확도 계산 생략하거나, 추후 평가에서 확인.
            if outputs.size(0) == targets.size(0):
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            else:
                # ESAM subset case: approximate or skip
                pass 
        else:
            # Mixup 시에는 정확도 계산이 애매하므로 패스하거나, soft label accuracy 계산
            pass
    
    avg_loss = total_loss / total
    avg_grad_norm = total_grad_norm / len(dataloader)
    # Mixup 사용 시 정확도는 Evaluate에서 확인하는 것이 정확함
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy, avg_grad_norm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # 평가 시에는 AMP를 쓰지 않거나, 써도 됨 (보통 inference 속도 향상을 위해 씀)
    # 여기서는 안전하게 FP32로 하거나 autocast 감싸도 됨.
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Evaluate는 보통 AMP 써도 되지만, 정확한 Loss 측정을 위해 FP32 권장되기도 함.
            # 속도를 위해 켬.
            with autocast('cuda', enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy