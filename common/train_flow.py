import random
import numpy as np
import torch
import torch.nn as nn
from common.optimizer import SAM, ESAM
from common.augmentation import rand_bbox, mixup_cutmix_data, mixup_cutmix_criterion
from torch.amp import autocast, GradScaler

import math


def get_total_grad_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of all parameter gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def sam_step(model, optimizer, criterion, inputs, label_info, mixup_applied, scaler):
    """Run a single SAM two-step update and return (loss, grad_norm, outputs)."""
    # first step: perturb weights
    with autocast('cuda', enabled=scaler is not None):
        outputs = model(inputs)
        loss = mixup_cutmix_criterion(criterion, outputs, *label_info) if mixup_applied else criterion(outputs, *label_info)

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

    # second step: compute gradient at perturbed weights and update
    with autocast('cuda', enabled=scaler is not None):
        outputs2 = model(inputs)
        second_loss = mixup_cutmix_criterion(criterion, outputs2, *label_info) if mixup_applied else criterion(outputs2, *label_info)

    if scaler:
        scaler.scale(second_loss).backward()

        inv_scale = 1.0 / scaler.get_scale()
        found_inf = False
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(inv_scale)
                if not torch.isfinite(p.grad).all():
                    found_inf = True

        if found_inf:
            # restore original weights if gradient contains inf/nan
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
    """Run a single ESAM two-step update and return (loss, grad_norm, outputs).

    The second step is performed only on the top-gamma fraction of samples
    ranked by loss increase after the first perturbation step.
    """
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    # first step: compute per-sample losses and perturb
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

    # select the top-gamma samples by loss increase
    with torch.no_grad():
        with autocast('cuda', enabled=scaler is not None):
            perturbed_outputs = model(inputs)
            if mixup_applied:
                perturbed_losses = lam * criterion_none(perturbed_outputs, targets_a) + (1 - lam) * criterion_none(perturbed_outputs, targets_b)
            else:
                perturbed_losses = criterion_none(perturbed_outputs, targets)

    loss_increases = perturbed_losses - base_losses
    _, indices = torch.topk(loss_increases, int(gamma * inputs.size(0)))

    # second step on selected subset
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
    """Run a standard (AdamW/SGD) optimizer step and return (loss, grad_norm, outputs)."""
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
    """Dispatch to the appropriate optimizer step based on optimizer type."""
    if isinstance(optimizer, SAM):
        return sam_step(model, optimizer, criterion, inputs, label_info, mixup_applied, scaler)
    elif isinstance(optimizer, ESAM):
        return esam_step(model, optimizer, criterion, inputs, label_info, mixup_applied, gamma, scaler)
    else:
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
    use_amp: bool = True
):
    """Train for one epoch and return (avg_loss, accuracy, avg_grad_norm).

    Accuracy is computed only when Mixup/CutMix is not applied and the output
    size matches the target size (e.g. ESAM returns a subset output).
    """
    model.train()
    total_loss, total_grad_norm, correct, total = 0.0, 0.0, 0, 0

    scaler = GradScaler() if use_amp else None

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        final_inputs, label_info, mixup_applied = mixup_cutmix_data(
            inputs, targets, use_mixup, use_cutmix, mixup_alpha
        )

        loss, grad_norm, outputs = update_step(
            model, optimizer, criterion, final_inputs, label_info, mixup_applied, gamma, scaler
        )

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        total_grad_norm += grad_norm

        if not mixup_applied:
            # ESAM returns a subset output, so only compute accuracy when sizes match
            if outputs.size(0) == targets.size(0):
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    avg_grad_norm = total_grad_norm / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0.0

    return avg_loss, accuracy, avg_grad_norm


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model and return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
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