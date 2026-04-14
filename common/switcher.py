import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from common.optimizer import SAM
from collections import deque
from dataclasses import dataclass


@dataclass
class DynamicSwitcher:
    """Signal-based switcher that monitors training statistics and triggers an
    AdamW to SAM transition when overfitting or stagnation is detected.

    The switch fires when a long-term plateau is observed, or when a short-term
    plateau coincides with a widening generalization gap or unstable grad norms.
    """

    def __init__(
        self,
        beta_ema: float = 0.95,
        history_window: int = 5,
        plateau_patience: int = 5,
        long_term_plateau_patience: int = 15,
        plateau_min_delta: float = 0.003,
        gap_increase_threshold: float = 0.015,
        grad_norm_increase_threshold: float = 0.07,
        min_switch_epoch: int = 15,
        oscillation_threshold: int = 3,
        enable_finetune: bool = True,
    ):
        """
        beta_ema                    : smoothing factor for EMA computation
        history_window              : number of past epochs used for trend detection
        plateau_patience            : consecutive epochs without improvement to declare a plateau
        long_term_plateau_patience  : patience for unconditional switch on extended plateau
        plateau_min_delta           : minimum val_acc improvement to reset the patience counter
        gap_increase_threshold      : EMA generalization gap increase considered significant
        grad_norm_increase_threshold: relative grad norm increase (as a fraction) considered significant
        min_switch_epoch            : earliest epoch the switcher may fire
        oscillation_threshold       : number of sign changes in grad norm EMA to declare oscillation
        """
        self.beta_ema = beta_ema
        self.history_window = history_window
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self.long_term_plateau_patience = long_term_plateau_patience
        self.gap_threshold = gap_increase_threshold
        self.grad_norm_threshold = grad_norm_increase_threshold
        self.min_switch_epoch = min_switch_epoch
        self.oscillation_threshold = oscillation_threshold

        self.history = deque(maxlen=history_window + 1)
        self.ema_values = {}
        self.patience_counter = 0
        self.best_val_acc = 0.0
        self.grad_norm_diff_history = deque(maxlen=history_window)

        self.switched = False
        self.enable_finetune = enable_finetune

    def __str__(self):
        return (
            f"beta_ema: {self.beta_ema}\n"
            f"history_window: {self.history_window}\n"
            f"plateau_patience: {self.plateau_patience}\n"
            f"plateau_min_delta: {self.plateau_min_delta}\n"
            f"gap_threshold: {self.gap_threshold}\n"
            f"grad_norm_increase_threshold: {self.grad_norm_threshold}\n"
            f"min_switch_epoch: {self.min_switch_epoch}\n"
            f"oscillation_threshold: {self.oscillation_threshold}"
        )

    def _update_ema(self, key: str, value: float):
        if key not in self.ema_values:
            self.ema_values[key] = value
        else:
            self.ema_values[key] = self.beta_ema * self.ema_values[key] + (1 - self.beta_ema) * value

    def step(self, epoch: int, train_acc: float, val_acc: float, grad_norm: float) -> bool:
        """Call once per epoch. Returns True when a switch should be triggered.

        epoch      : current epoch (0-indexed)
        train_acc  : training accuracy
        val_acc    : validation accuracy
        grad_norm  : average gradient norm for the epoch
        """
        self._update_ema('train_acc', train_acc)
        self._update_ema('val_acc', val_acc)
        self._update_ema('grad_norm', grad_norm)

        ema_gap = self.ema_values['train_acc'] - self.ema_values['val_acc']
        current_state = {
            'ema_gap': ema_gap,
            'ema_grad_norm': self.ema_values['grad_norm']
        }
        self.history.append(current_state)

        if epoch < self.min_switch_epoch or len(self.history) <= self.history_window:
            print(f"[Switcher] Warming up... (Epoch {epoch+1})")
            return False

        if val_acc > self.best_val_acc + self.plateau_min_delta:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        is_plateau = self.patience_counter >= self.plateau_patience
        is_long_term_plateau = self.patience_counter >= self.long_term_plateau_patience

        gap_current = self.history[-1]['ema_gap']
        gap_historical = self.history[0]['ema_gap']
        is_gap_increasing = (gap_current - gap_historical) > self.gap_threshold

        grad_norm_current = self.history[-1]['ema_grad_norm']
        grad_norm_historical = self.history[0]['ema_grad_norm']

        if grad_norm_historical < 1e-6:
            grad_norm_historical = 1e-6
        relative_increase = (grad_norm_current / grad_norm_historical) - 1
        is_grad_norm_increasing = relative_increase > self.grad_norm_threshold

        is_oscillating = False
        sign_changes = 0
        if len(self.history) > 1:
            current_grad_norm_ema = self.history[-1]['ema_grad_norm']
            prev_grad_norm_ema = self.history[-2]['ema_grad_norm']
            diff = current_grad_norm_ema - prev_grad_norm_ema
            if diff != 0:
                self.grad_norm_diff_history.append(np.sign(diff))

        if len(self.grad_norm_diff_history) >= self.history_window:
            for i in range(len(self.grad_norm_diff_history) - 1):
                if self.grad_norm_diff_history[i] != self.grad_norm_diff_history[i+1]:
                    sign_changes += 1
            if sign_changes >= self.oscillation_threshold:
                is_oscillating = True

        print(
            f"[Switcher] Status | Plateau: {is_plateau} ({self.patience_counter}/{self.plateau_patience}) | "
            f"Gap Increase: {is_gap_increasing} ({gap_current - gap_historical:.3f} > {self.gap_threshold}) | "
            f"Grad Norm Increase: {is_grad_norm_increasing} ({relative_increase:.3f} > {self.grad_norm_threshold}) | "
            f"Oscillate: {is_oscillating} ({sign_changes}/{self.oscillation_threshold})"
        )

        if is_long_term_plateau or (is_plateau and (is_gap_increasing or is_grad_norm_increasing or is_oscillating)):
            return True

        return False


@dataclass
class DynamicSwitcher_ver02:
    """Improved switcher using necessary and sufficient condition separation.

    Necessary conditions (both must hold before sufficient conditions are evaluated):
      1. epoch >= min_switch_epoch
      2. train loss has entered a stable region (std < threshold)

    Sufficient conditions (evaluated as a weighted score; switch when score >= threshold):
      1. val_acc improvement slope has slowed (most direct signal)
      2. val_acc plateau
      3. widening generalization gap
      4. unstable or growing grad norm

    Compared to DynamicSwitcher:
      - Separating necessary/sufficient conditions reduces false positives.
      - Weighted score replaces binary OR, making the decision more noise-robust.
      - Slope signal detects stagnation earlier and more precisely than patience alone.
      - Loss stability check prevents switching during unstable training.
    """

    def __init__(
        self,
        # necessary condition parameters
        min_switch_epoch: int = 150,
        loss_stable_window: int = 10,
        loss_stable_std_threshold: float = 0.02,

        # sufficient condition: improvement rate
        slope_window: int = 30,
        slope_threshold: float = 0.01,

        # sufficient condition: auxiliary signals
        beta_ema: float = 0.9,
        history_window: int = 20,
        plateau_patience: int = 20,
        plateau_min_delta: float = 0.005,
        gap_increase_threshold: float = 0.02,
        grad_norm_increase_threshold: float = 0.1,
        oscillation_threshold: int = 5,

        # score weights (should sum to 1)
        w_slope: float = 0.40,    # improvement slowdown (most direct)
        w_plateau: float = 0.25,  # val_acc plateau
        w_gap: float = 0.20,      # generalization gap increase
        w_grad: float = 0.15,     # grad norm instability

        # switch trigger threshold
        score_threshold: float = 0.5,
    ):
        # necessary conditions
        self.min_switch_epoch = min_switch_epoch
        self.loss_stable_window = loss_stable_window
        self.loss_stable_std_threshold = loss_stable_std_threshold

        # sufficient condition: slope
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold

        # sufficient condition: auxiliary signals
        self.beta_ema = beta_ema
        self.history_window = history_window
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self.gap_threshold = gap_increase_threshold
        self.grad_norm_threshold = grad_norm_increase_threshold
        self.oscillation_threshold = oscillation_threshold

        # score weights
        self.w_slope = w_slope
        self.w_plateau = w_plateau
        self.w_gap = w_gap
        self.w_grad = w_grad
        self.score_threshold = score_threshold

        # internal state
        self.val_acc_history = deque(maxlen=max(slope_window + 1, history_window + 1))
        self.train_loss_history = deque(maxlen=loss_stable_window)
        self.ema_values = {}
        self.grad_norm_diff_history = deque(maxlen=history_window)
        self.ema_history = deque(maxlen=history_window + 1)

        self.patience_counter = 0
        self.best_val_acc = 0.0

    def _update_ema(self, key: str, value: float):
        if key not in self.ema_values:
            self.ema_values[key] = value
        else:
            self.ema_values[key] = (
                self.beta_ema * self.ema_values[key]
                + (1 - self.beta_ema) * value
            )

    def _check_min_epoch(self, epoch: int) -> bool:
        """Necessary condition 1: epoch has passed the minimum threshold."""
        return epoch >= self.min_switch_epoch

    def _check_loss_stable(self) -> tuple[bool, float]:
        """Necessary condition 2: train loss std is below the stability threshold."""
        if len(self.train_loss_history) < self.loss_stable_window:
            return False, float('inf')
        loss_std = float(np.std(list(self.train_loss_history)))
        return loss_std < self.loss_stable_std_threshold, loss_std

    def _check_slope(self) -> tuple[float, float]:
        """Sufficient condition 1: val_acc improvement rate over slope_window epochs.

        Returns a normalized score in [0, 1] and the actual slope value.
        Score is 0 when slope >= threshold, and increases linearly toward 1 as slope approaches 0.
        """
        if len(self.val_acc_history) < self.slope_window + 1:
            return 0.0, float('inf')

        history_list = list(self.val_acc_history)
        recent = history_list[-self.slope_window - 1:]
        x = np.arange(len(recent))
        slope = float(np.polyfit(x, recent, 1)[0])  # least-squares linear slope

        if slope >= self.slope_threshold:
            score = 0.0
        else:
            # linearly scale from threshold down to 0; below 0 clamps to 1.0
            score = min(1.0, (self.slope_threshold - slope) / self.slope_threshold)

        return score, slope

    def _check_plateau(self) -> tuple[float, int]:
        """Sufficient condition 2: patience counter as a fraction of plateau_patience."""
        score = min(1.0, self.patience_counter / self.plateau_patience)
        return score, self.patience_counter

    def _check_gap(self) -> tuple[float, float]:
        """Sufficient condition 3: increase in EMA generalization gap over the history window."""
        if len(self.ema_history) <= self.history_window:
            return 0.0, 0.0
        gap_current = self.ema_history[-1]['ema_gap']
        gap_historical = self.ema_history[0]['ema_gap']
        gap_delta = gap_current - gap_historical

        if gap_delta <= 0:
            score = 0.0
        else:
            # normalize against 2x the threshold
            score = min(1.0, gap_delta / (self.gap_threshold * 2))
        return score, gap_delta

    def _check_grad_norm(self) -> tuple[float, float]:
        """Sufficient condition 4: grad norm instability (increase or oscillation)."""
        if len(self.ema_history) <= self.history_window:
            return 0.0, 0.0
        grad_current = self.ema_history[-1]['ema_grad_norm']
        grad_historical = self.ema_history[0]['ema_grad_norm']

        if grad_historical < 1e-6:
            grad_historical = 1e-6
        relative_increase = (grad_current / grad_historical) - 1.0

        sign_changes = 0
        if len(self.grad_norm_diff_history) >= self.history_window:
            diffs = list(self.grad_norm_diff_history)
            for i in range(len(diffs) - 1):
                if diffs[i] != diffs[i + 1]:
                    sign_changes += 1

        is_oscillating = sign_changes >= self.oscillation_threshold
        is_increasing = relative_increase > self.grad_norm_threshold

        if is_oscillating and is_increasing:
            score = 1.0
        elif is_oscillating or is_increasing:
            score = 0.6
        else:
            score = 0.0

        return score, relative_increase

    def step(
        self,
        epoch: int,
        train_acc: float,
        val_acc: float,
        train_loss: float,
        grad_norm: float,
    ) -> bool:
        """Call once per epoch. Returns True when a switch should be triggered.

        epoch      : current epoch (0-indexed)
        train_acc  : training accuracy (0 to 1)
        val_acc    : validation accuracy (0 to 1)
        train_loss : training loss for loss stability check
        grad_norm  : average gradient norm
        """
        self.val_acc_history.append(val_acc)
        self.train_loss_history.append(train_loss)

        self._update_ema('train_acc', train_acc)
        self._update_ema('val_acc', val_acc)
        self._update_ema('grad_norm', grad_norm)

        ema_gap = self.ema_values['train_acc'] - self.ema_values['val_acc']
        current_state = {
            'ema_gap': ema_gap,
            'ema_grad_norm': self.ema_values['grad_norm']
        }
        self.ema_history.append(current_state)

        if len(self.ema_history) > 1:
            diff = self.ema_history[-1]['ema_grad_norm'] - self.ema_history[-2]['ema_grad_norm']
            if diff != 0:
                self.grad_norm_diff_history.append(np.sign(diff))

        if val_acc > self.best_val_acc + self.plateau_min_delta:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # evaluate necessary conditions first
        ok_epoch = self._check_min_epoch(epoch)
        ok_stable, loss_std = self._check_loss_stable()

        if not ok_epoch:
            print(f"[Switcher] Warming up... (epoch {epoch+1}/{self.min_switch_epoch})")
            return False

        if not ok_stable:
            print(
                f"[Switcher] Waiting for loss stability "
                f"(std={loss_std:.4f} > {self.loss_stable_std_threshold})"
            )
            return False

        # evaluate sufficient conditions and compute weighted score
        s_slope, slope = self._check_slope()
        s_plateau, p_count = self._check_plateau()
        s_gap, gap_delta = self._check_gap()
        s_grad, grad_rel = self._check_grad_norm()

        score = (
            self.w_slope * s_slope
            + self.w_plateau * s_plateau
            + self.w_gap * s_gap
            + self.w_grad * s_grad
        )

        print(
            f"[Switcher] Score: {score:.3f}/{self.score_threshold:.2f} | "
            f"slope={slope:.4f}%/ep(s={s_slope:.2f}) | "
            f"plateau={p_count}/{self.plateau_patience}(s={s_plateau:.2f}) | "
            f"gap_delta={gap_delta:.3f}(s={s_gap:.2f}) | "
            f"grad_rel={grad_rel:.3f}(s={s_grad:.2f})"
        )

        if score >= self.score_threshold:
            print(f"[Switcher] Threshold reached! Triggering switch.")
            return True

        return False


class DynamicSwitcher_ver03:
    """Minimal slope-only switcher.

    Switch condition:
        epoch >= min_switch_epoch AND val_acc slope < slope_threshold (%/epoch)

    Simpler than ver02 and suitable for cases where a single clear signal suffices.
    """

    def __init__(
        self,
        min_switch_epoch: int = 150,
        slope_window: int = 30,
        slope_threshold: float = 0.01,
    ):
        """
        min_switch_epoch : earliest epoch the switcher may fire.
                           Prevents premature switching before AdamW has converged.
                           Recommended: 40-60% of total epochs.
        slope_window     : number of epochs over which to measure val_acc improvement rate.
                           Too short makes it noise-sensitive; too long delays the response.
                           Recommended: 20-40.
        slope_threshold  : improvement rate (%/epoch) below which a switch is triggered.
                           Values near 0 mean the switch fires only when AdamW has fully stalled.
                           Recommended: 0.005-0.02.
        """
        self.min_switch_epoch = min_switch_epoch
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold

        self.val_acc_history = deque(maxlen=slope_window + 1)

    def step(self, epoch: int, val_acc: float) -> bool:
        """Call once per epoch. Returns True when a switch should be triggered.

        epoch   : current epoch (0-indexed)
        val_acc : validation accuracy (use a consistent scale, e.g. 0-100 or 0-1)
        """
        self.val_acc_history.append(val_acc)

        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False

        if len(self.val_acc_history) < self.slope_window + 1:
            print(f"[Switcher] Collecting slope data... "
                  f"({len(self.val_acc_history)}/{self.slope_window + 1})")
            return False

        history = list(self.val_acc_history)
        x = np.arange(len(history))
        slope = float(np.polyfit(x, history, 1)[0])  # least-squares linear slope

        print(f"[Switcher] slope={slope:.4f} %/ep (threshold={self.slope_threshold})")

        if slope < self.slope_threshold:
            print(f"[Switcher] Slope below threshold. Switch triggered!")
            return True

        return False


class DynamicSwitcher_ver04:
    """Nesterov-style switcher using LR restart.

    Runs short forward simulations of both SAM and AdamW on a probe subset of
    the validation set and switches to SAM when the predicted accuracy gain exceeds
    gain_threshold. Uses restart_lr = initial_lr * lr_restart_factor for the
    simulation and for the actual switch, ensuring consistency.

    LR restart is necessary because cosine-decayed LR values are often too small
    for SAM perturbation to meaningfully explore flat minima.

    Recommended lr_restart_factor: 0.1 to 0.5.
    """

    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 10,
        probe_ratio: float = 0.1,
        sim_steps: int = 10,
        gain_threshold: float = 0.2,
        rho: float = 0.05,
        weight_decay: float = 0.05,
        initial_lr: float = 0.001,
        lr_restart_factor: float = 0.3,
    ):
        self.min_switch_epoch = min_switch_epoch
        self.check_every = check_every
        self.probe_ratio = probe_ratio
        self.sim_steps = sim_steps
        self.gain_threshold = gain_threshold
        self.rho = rho
        self.weight_decay = weight_decay
        self.initial_lr = initial_lr
        self.lr_restart_factor = lr_restart_factor

        # LR used both in simulation and in the actual switch
        self.restart_lr = initial_lr * lr_restart_factor

    def _make_probe_loader(self, val_loader: DataLoader) -> DataLoader:
        """Sample a random subset of the validation set for simulation."""
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )

    @torch.no_grad()
    def _probe_accuracy(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """Evaluate accuracy on the probe loader without gradient computation."""
        model.eval()
        correct, total = 0, 0
        for inputs, targets in probe_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def _simulate_sam(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        adamw_state: dict,
    ) -> float:
        """Simulate SAM(AdamW base) for sim_steps using restart_lr.

        AdamW momentum state is transferred to match actual post-switch conditions.
        """
        from common.optimizer import SAM

        sim_sam = SAM(
            model.parameters(),
            optim.AdamW,
            rho=self.rho,
            lr=self.restart_lr,  # use restart_lr rather than the decayed AdamW LR
            weight_decay=self.weight_decay,
        )

        # transfer AdamW momentum/variance state to match actual switch conditions
        param_list = list(model.parameters())
        for i, param in enumerate(param_list):
            if i in adamw_state:
                sim_sam.base_optimizer.state[param] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in adamw_state[i].items()
                }

        model.train()
        probe_iter = iter(probe_loader)
        for _ in range(self.sim_steps):
            try:
                inputs, targets = next(probe_iter)
            except StopIteration:
                probe_iter = iter(probe_loader)
                inputs, targets = next(probe_iter)

            inputs, targets = inputs.to(device), targets.to(device)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_sam.first_step(zero_grad=True)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_sam.second_step(zero_grad=True)

        return self._probe_accuracy(model, probe_loader, criterion, device)

    def _simulate_adamw(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        current_lr: float,
        adamw_state: dict,
    ) -> float:
        """Simulate AdamW for sim_steps using the current (cosine-decayed) LR.

        Represents the baseline "what happens if we keep using AdamW as-is".
        """
        sim_adamw = optim.AdamW(
            model.parameters(),
            lr=current_lr,
            weight_decay=self.weight_decay,
        )
        try:
            sim_adamw.load_state_dict({
                'state': adamw_state,
                'param_groups': sim_adamw.state_dict()['param_groups'],
            })
        except Exception:
            pass

        model.train()
        probe_iter = iter(probe_loader)
        for _ in range(self.sim_steps):
            try:
                inputs, targets = next(probe_iter)
            except StopIteration:
                probe_iter = iter(probe_loader)
                inputs, targets = next(probe_iter)

            inputs, targets = inputs.to(device), targets.to(device)
            sim_adamw.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_adamw.step()

        return self._probe_accuracy(model, probe_loader, criterion, device)

    def step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> bool:
        """Call once per epoch. Returns True when a switch should be triggered."""
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False

        elapsed = epoch - self.min_switch_epoch
        if elapsed > 0 and elapsed % self.check_every != 0:
            remaining = self.check_every - (elapsed % self.check_every)
            print(f"[Switcher] Skipping simulation (next check in {remaining} epoch)")
            return False

        current_lr = optimizer.param_groups[0]['lr']
        adamw_state = deepcopy(optimizer.state_dict()['state'])
        w_saved = deepcopy(model.state_dict())
        probe_loader = self._make_probe_loader(val_loader)

        print(f"\n[Switcher] Running simulation at epoch {epoch + 1}")
        print(f"  current_lr={current_lr:.6f}  restart_lr={self.restart_lr:.6f}"
              f"  rho={self.rho}  sim_steps={self.sim_steps}"
              f"  probe={self.probe_ratio*100:.0f}% of val")

        acc_sam = self._simulate_sam(model, probe_loader, criterion, device, adamw_state)
        model.load_state_dict(w_saved)

        acc_adamw = self._simulate_adamw(model, probe_loader, criterion, device, current_lr, adamw_state)
        model.load_state_dict(w_saved)

        predicted_gain = acc_sam - acc_adamw
        print(f"  SAM(restart)={acc_sam:.2f}%  AdamW(current)={acc_adamw:.2f}%"
              f"  gain={predicted_gain:+.2f}%  (threshold={self.gain_threshold:+.2f}%)")

        if predicted_gain >= self.gain_threshold:
            print(f"  Gain exceeds threshold. Switch triggered!\n")
            return True

        print(f"  Gain insufficient. Keep AdamW\n")
        return False


class DynamicSwitcher_ver05:
    """Nesterov-style switcher with two key improvements over ver04.

    Change 1: fair simulation comparison.
        ver04 compared SAM(restart_lr) vs AdamW(current_lr). When the cosine LR
        is nearly exhausted, AdamW barely moves and SAM wins not because of its
        perturbation but because of the LR advantage. ver05 uses restart_lr for
        both, so the measured gain reflects SAM perturbation itself.

    Change 2: separate switch_rho for post-switch SAM.
        ver04 used the same rho for simulation and the actual switch. Converged
        weights are sensitive to large perturbations, so switch_rho is introduced
        as a separate (smaller) value (e.g. 0.05) for the post-switch phase.
    """

    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 10,
        probe_ratio: float = 0.1,
        sim_steps: int = 10,
        gain_threshold: float = 0.2,
        switch_rho: float = 0.05,   # separate rho for post-switch SAM (ver05 change)
        weight_decay: float = 0.05,
        initial_lr: float = 0.001,
        lr_restart_factor: float = 0.3,
    ):
        self.min_switch_epoch = min_switch_epoch
        self.check_every = check_every
        self.probe_ratio = probe_ratio
        self.sim_steps = sim_steps
        self.gain_threshold = gain_threshold
        self.switch_rho = switch_rho
        self.weight_decay = weight_decay
        self.initial_lr = initial_lr
        self.lr_restart_factor = lr_restart_factor
        self.restart_lr = initial_lr * lr_restart_factor

    def _make_probe_loader(self, val_loader: DataLoader) -> DataLoader:
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )

    @torch.no_grad()
    def _probe_accuracy(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        model.eval()
        correct, total = 0, 0
        for inputs, targets in probe_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def _simulate_sam(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        adamw_state: dict,
    ) -> float:
        """Simulate SAM with switch_rho and restart_lr."""
        from common.optimizer import SAM

        sim_sam = SAM(
            model.parameters(),
            optim.AdamW,
            rho=self.switch_rho,  # use switch_rho instead of rho (ver05 change)
            lr=self.restart_lr,
            weight_decay=self.weight_decay,
        )

        param_list = list(model.parameters())
        for i, param in enumerate(param_list):
            if i in adamw_state:
                sim_sam.base_optimizer.state[param] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in adamw_state[i].items()
                }

        model.train()
        probe_iter = iter(probe_loader)
        for _ in range(self.sim_steps):
            try:
                inputs, targets = next(probe_iter)
            except StopIteration:
                probe_iter = iter(probe_loader)
                inputs, targets = next(probe_iter)

            inputs, targets = inputs.to(device), targets.to(device)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_sam.first_step(zero_grad=True)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_sam.second_step(zero_grad=True)

        return self._probe_accuracy(model, probe_loader, criterion, device)

    def _simulate_adamw(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        adamw_state: dict,
    ) -> float:
        """Simulate AdamW at restart_lr for a fair comparison against SAM.

        ver04 used the current (decayed) LR for AdamW, which made SAM appear
        better simply due to the LR difference. Using restart_lr here isolates
        the effect of SAM perturbation.
        """
        sim_adamw = optim.AdamW(
            model.parameters(),
            lr=self.restart_lr,  # changed from current_lr to restart_lr (ver05 change)
            weight_decay=self.weight_decay,
        )

        param_list = list(model.parameters())
        for i, param in enumerate(param_list):
            if i in adamw_state:
                sim_adamw.state[param] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in adamw_state[i].items()
                }

        model.train()
        probe_iter = iter(probe_loader)
        for _ in range(self.sim_steps):
            try:
                inputs, targets = next(probe_iter)
            except StopIteration:
                probe_iter = iter(probe_loader)
                inputs, targets = next(probe_iter)

            inputs, targets = inputs.to(device), targets.to(device)
            sim_adamw.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_adamw.step()

        return self._probe_accuracy(model, probe_loader, criterion, device)

    def step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> bool:
        """Call once per epoch. Returns True when a switch should be triggered."""
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False

        elapsed = epoch - self.min_switch_epoch
        if elapsed > 0 and elapsed % self.check_every != 0:
            remaining = self.check_every - (elapsed % self.check_every)
            print(f"[Switcher] Skipping simulation (next check in {remaining} epoch)")
            return False

        current_lr = optimizer.param_groups[0]['lr']
        adamw_state = deepcopy(optimizer.state_dict()['state'])
        w_saved = deepcopy(model.state_dict())
        probe_loader = self._make_probe_loader(val_loader)

        print(f"\n[Switcher] Running simulation at epoch {epoch + 1}")
        print(f"  current_lr={current_lr:.6f}  restart_lr={self.restart_lr:.6f}"
              f"  switch_rho={self.switch_rho}  sim_steps={self.sim_steps}"
              f"  probe={self.probe_ratio*100:.0f}% of val")
        print(f"  Both SAM and AdamW simulated at restart_lr={self.restart_lr:.6f}")

        acc_sam = self._simulate_sam(model, probe_loader, criterion, device, adamw_state)
        model.load_state_dict(w_saved)

        acc_adamw = self._simulate_adamw(model, probe_loader, criterion, device, adamw_state)
        model.load_state_dict(w_saved)

        predicted_gain = acc_sam - acc_adamw
        print(f"  SAM(restart)={acc_sam:.2f}%  AdamW(restart)={acc_adamw:.2f}%"
              f"  gain={predicted_gain:+.2f}%  (threshold={self.gain_threshold:+.2f}%)")

        if predicted_gain >= self.gain_threshold:
            print(f"  Gain exceeds threshold. Switch triggered!\n")
            return True

        print(f"  Gain insufficient. Keep AdamW\n")
        return False


class DynamicSwitcher_ver06:
    """Nesterov-style switcher with stronger simulation reliability over ver05.

    Change 1: stronger simulation reliability.
        probe_ratio: 0.1 -> 0.3 (more samples reduce variance).
        sim_steps: 10 -> 20 (steps past the initial loss spike).
        gain_threshold: 0.2 -> 1.0 (higher bar prevents noise-driven switches).

    Change 2: simulate at rho_max rather than switch_rho.
        ver05 used switch_rho=0.05 which is too small to show SAM's benefit.
        ver06 uses rho_max (the target rho after warmup) for the simulation,
        directly measuring whether SAM at its intended operating rho is beneficial.
        The rho warmup itself is handled by the main training script.
    """

    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 10,
        probe_ratio: float = 0.3,   # increased from 0.1
        sim_steps: int = 20,        # increased from 10
        gain_threshold: float = 1.0,  # increased from 0.2
        rho_max: float = 0.15,      # target rho after warmup; used in simulation
        weight_decay: float = 0.05,
        initial_lr: float = 0.001,
        lr_restart_factor: float = 0.3,
    ):
        self.min_switch_epoch = min_switch_epoch
        self.check_every = check_every
        self.probe_ratio = probe_ratio
        self.sim_steps = sim_steps
        self.gain_threshold = gain_threshold
        self.rho_max = rho_max
        self.weight_decay = weight_decay
        self.initial_lr = initial_lr
        self.lr_restart_factor = lr_restart_factor
        self.restart_lr = initial_lr * lr_restart_factor

    def _make_probe_loader(self, val_loader: DataLoader) -> DataLoader:
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )

    @torch.no_grad()
    def _probe_accuracy(
        self, model: nn.Module, probe_loader: DataLoader,
        criterion: nn.Module, device: torch.device,
    ) -> float:
        model.eval()
        correct, total = 0, 0
        for inputs, targets in probe_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def _simulate_sam(
        self, model: nn.Module, probe_loader: DataLoader,
        criterion: nn.Module, device: torch.device,
        adamw_state: dict,
    ) -> float:
        """Simulate SAM at rho_max (the intended operating rho after warmup)."""
        from common.optimizer import SAM

        sim_sam = SAM(
            model.parameters(), optim.AdamW,
            rho=self.rho_max,   # simulate at the target rho_max, not a conservative smaller value
            lr=self.restart_lr,
            weight_decay=self.weight_decay,
        )

        param_list = list(model.parameters())
        for i, param in enumerate(param_list):
            if i in adamw_state:
                sim_sam.base_optimizer.state[param] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in adamw_state[i].items()
                }

        model.train()
        probe_iter = iter(probe_loader)
        for _ in range(self.sim_steps):
            try:
                inputs, targets = next(probe_iter)
            except StopIteration:
                probe_iter = iter(probe_loader)
                inputs, targets = next(probe_iter)

            inputs, targets = inputs.to(device), targets.to(device)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_sam.first_step(zero_grad=True)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_sam.second_step(zero_grad=True)

        return self._probe_accuracy(model, probe_loader, criterion, device)

    def _simulate_adamw(
        self, model: nn.Module, probe_loader: DataLoader,
        criterion: nn.Module, device: torch.device,
        adamw_state: dict,
    ) -> float:
        """Simulate AdamW at restart_lr for a fair comparison (same as ver05)."""
        sim_adamw = optim.AdamW(
            model.parameters(),
            lr=self.restart_lr,
            weight_decay=self.weight_decay,
        )

        param_list = list(model.parameters())
        for i, param in enumerate(param_list):
            if i in adamw_state:
                sim_adamw.state[param] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in adamw_state[i].items()
                }

        model.train()
        probe_iter = iter(probe_loader)
        for _ in range(self.sim_steps):
            try:
                inputs, targets = next(probe_iter)
            except StopIteration:
                probe_iter = iter(probe_loader)
                inputs, targets = next(probe_iter)

            inputs, targets = inputs.to(device), targets.to(device)
            sim_adamw.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            sim_adamw.step()

        return self._probe_accuracy(model, probe_loader, criterion, device)

    def step(
        self, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
        val_loader: DataLoader, criterion: nn.Module, device: torch.device,
    ) -> bool:
        """Call once per epoch. Returns True when a switch should be triggered."""
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False

        elapsed = epoch - self.min_switch_epoch
        if elapsed > 0 and elapsed % self.check_every != 0:
            remaining = self.check_every - (elapsed % self.check_every)
            print(f"[Switcher] Skipping simulation (next check in {remaining} epoch)")
            return False

        current_lr = optimizer.param_groups[0]['lr']
        adamw_state = deepcopy(optimizer.state_dict()['state'])
        w_saved = deepcopy(model.state_dict())
        probe_loader = self._make_probe_loader(val_loader)

        print(f"\n[Switcher] Running simulation at epoch {epoch + 1}")
        print(f"  current_lr={current_lr:.6f}  restart_lr={self.restart_lr:.6f}"
              f"  rho_max={self.rho_max}  sim_steps={self.sim_steps}"
              f"  probe={self.probe_ratio*100:.0f}% of val")
        print(f"  Both at restart_lr={self.restart_lr:.6f}, SAM uses rho_max={self.rho_max}")

        acc_sam = self._simulate_sam(model, probe_loader, criterion, device, adamw_state)
        model.load_state_dict(w_saved)

        acc_adamw = self._simulate_adamw(model, probe_loader, criterion, device, adamw_state)
        model.load_state_dict(w_saved)

        predicted_gain = acc_sam - acc_adamw
        print(f"  SAM(rho={self.rho_max})={acc_sam:.2f}%"
              f"  AdamW={acc_adamw:.2f}%"
              f"  gain={predicted_gain:+.2f}%  (threshold={self.gain_threshold:+.2f}%)")

        if predicted_gain >= self.gain_threshold:
            print(f"  Gain exceeds threshold. Switch triggered!\n")
            return True

        print(f"  Gain insufficient. Keep AdamW\n")
        return False


class SharpnessAwareSwitcher:
    """Sharpness-based switcher that replaces short-horizon accuracy simulation.

    Paradigm shift from ver04-06:
        Nesterov-style simulations compare accuracy N steps ahead, but SAM is
        intentionally slow in the short term. AdamW always wins in that window,
        so the question itself is flawed. This class changes the question:
        "Is the current loss landscape sharp?" rather than "Is SAM better now?"

    Sharpness measurement:
        sharpness = loss(w + rho * g/||g||) - loss(w)
        Same perturbation as SAM first_step. Requires 2 forward passes + 1 backward,
        making it roughly 10x cheaper than a 20-step simulation.

    Switch condition:
        epoch >= min_switch_epoch AND EMA(sharpness) >= sharpness_threshold

    Monitoring mode:
        Set sharpness_threshold to a large value (e.g. 999.0) to collect sharpness
        data without actually switching. Use the logged values to choose a threshold.
    """

    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 5,
        probe_ratio: float = 0.2,
        sharpness_rho: float = 0.15,
        sharpness_threshold: float = 0.5,
        sharpness_ema_beta: float = 0.9,
    ):
        """
        min_switch_epoch    : earliest epoch the switch may be considered
        check_every         : how often sharpness is measured (in epochs)
        probe_ratio         : fraction of val set used for the sharpness probe
        sharpness_rho       : perturbation size (recommend matching rho_max)
        sharpness_threshold : EMA sharpness above this value triggers the switch
        sharpness_ema_beta  : EMA smoothing factor
        """
        self.min_switch_epoch = min_switch_epoch
        self.check_every = check_every
        self.probe_ratio = probe_ratio
        self.sharpness_rho = sharpness_rho
        self.sharpness_threshold = sharpness_threshold
        self.ema_beta = sharpness_ema_beta

        self.sharpness_history = []
        self.ema_sharpness = None

    def _make_probe_loader(self, val_loader):
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )

    def _measure_sharpness(self, model, probe_loader, criterion, device):
        """Compute sharpness = loss(w + rho*g/||g||) - loss(w) on the probe set.

        Step 1: compute loss and accumulate gradients on original weights.
        Step 2: perturb weights in the gradient direction by rho.
        Step 3: compute loss on perturbed weights (no gradient).
        Step 4: restore original weights.
        """
        model.train()
        model.zero_grad()

        # step 1: original loss + gradient
        total_loss_orig = 0.0
        total_samples = 0
        for inputs, targets in probe_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            total_loss_orig += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss_orig = total_loss_orig / total_samples

        # step 2: compute grad norm and perturb weights
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if grad_norm < 1e-12:
            model.zero_grad()
            return 0.0

        old_params = {}
        scale = self.sharpness_rho / grad_norm
        for name, p in model.named_parameters():
            if p.grad is not None:
                old_params[name] = p.data.clone()
                p.data.add_(p.grad * scale)

        # step 3: perturbed loss (no grad)
        total_loss_pert = 0.0
        with torch.no_grad():
            for inputs, targets in probe_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                total_loss_pert += loss.item() * inputs.size(0)

        avg_loss_pert = total_loss_pert / total_samples

        # step 4: restore original weights
        for name, p in model.named_parameters():
            if name in old_params:
                p.data.copy_(old_params[name])
        model.zero_grad()

        return avg_loss_pert - avg_loss_orig

    def step(self, epoch, model, val_loader, criterion, device):
        """Call once per epoch. Returns True when a switch should be triggered.

        Sharpness is measured even before min_switch_epoch to collect baseline data.
        """
        if epoch % self.check_every != 0:
            if epoch < self.min_switch_epoch:
                print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            else:
                remaining = self.check_every - (epoch % self.check_every)
                print(f"[Switcher] Next sharpness check in {remaining} epoch")
            return False

        probe_loader = self._make_probe_loader(val_loader)
        sharpness = self._measure_sharpness(model, probe_loader, criterion, device)

        if self.ema_sharpness is None:
            self.ema_sharpness = sharpness
        else:
            self.ema_sharpness = (
                self.ema_beta * self.ema_sharpness
                + (1 - self.ema_beta) * sharpness
            )

        self.sharpness_history.append({
            'epoch': epoch + 1,
            'sharpness': sharpness,
            'ema_sharpness': self.ema_sharpness,
        })

        phase = "MONITOR" if epoch < self.min_switch_epoch else "EVALUATE"
        print(
            f"\n[Switcher] Sharpness at epoch {epoch + 1} [{phase}]"
            f"\n  raw={sharpness:.4f}  ema={self.ema_sharpness:.4f}"
            f"  threshold={self.sharpness_threshold}  rho={self.sharpness_rho}"
        )

        if epoch < self.min_switch_epoch:
            print(f"  Monitoring only ({epoch + 1}/{self.min_switch_epoch})")
            return False

        if self.ema_sharpness >= self.sharpness_threshold:
            print(f"  SWITCH TRIGGERED! ema={self.ema_sharpness:.4f} >= {self.sharpness_threshold}\n")
            return True

        print(f"  Keep AdamW (ema={self.ema_sharpness:.4f} < {self.sharpness_threshold})\n")
        return False