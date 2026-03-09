import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from common.optimizer import SAM
from collections import deque
from dataclasses import dataclass

@dataclass
class DynamicSwitcher:
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
        @beta_ema (float): EMA 계산을 위한 평활 계수 (smoothing factor)
        @history_window (int): 신호의 추세 판단을 위한 과거 에포크 윈도우 (H)
        @plateau_patience (int): 성능 정체(Plateau)를 판단하기 위한 연속 에포크 수
        @plateau_min_delta (float): 성능 향상으로 간주할 최소 변화량 (epsilon_p)
        @gap_increase_threshold (float): Gap 증가로 판단할 임계값 (gamma_g)
        @grad_norm_increase_threshold (float): Grad Norm 증가로 판단할 임계값 (gamma_n, %)
        @min_switch_epoch (int): 스위처가 작동을 시작하는 최소 에포크
        @oscillation_threshold (int): 진동으로 판단할 부호 변경 횟수 임계값
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
        """
        @epoch (int): 현재 에포크
        @train_acc (float): 현재 훈련 정확도
        @val_acc (float): 현재 검증 정확도
        @grad_norm (float): 현재 그래디언트 놈
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

        if grad_norm_historical < 1e-6: grad_norm_historical = 1e-6
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
        
        if is_long_term_plateau or (is_plateau and (is_gap_increasing or (is_grad_norm_increasing or is_oscillating))):
            return True
        
        return False