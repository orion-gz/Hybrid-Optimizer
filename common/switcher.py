import torch
import numpy as np
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.amp import autocast
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, Subset
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
    
    
@dataclass
class DynamicSwitcher_ver02:
    def __init__(
        self,
        # ── 필요조건 파라미터 ──────────────────────────────────────────
        min_switch_epoch: int = 150,
        loss_stable_window: int = 10,       # loss 안정성 판단 윈도우
        loss_stable_std_threshold: float = 0.02,  # loss std 이 이하면 안정
 
        # ── 충분조건: 개선 속도 ────────────────────────────────────────
        slope_window: int = 30,             # 개선 속도 측정 윈도우 (epoch)
        slope_threshold: float = 0.01,      # %/epoch 이하면 "개선 둔화"
 
        # ── 충분조건: 기존 신호 (보조) ────────────────────────────────
        beta_ema: float = 0.9,
        history_window: int = 20,
        plateau_patience: int = 20,
        plateau_min_delta: float = 0.005,
        gap_increase_threshold: float = 0.02,
        grad_norm_increase_threshold: float = 0.1,
        oscillation_threshold: int = 5,
 
        # ── 점수 가중치 ────────────────────────────────────────────────
        # 각 충분조건 신호의 기여 가중치 (합이 1이 되도록 설계)
        w_slope: float = 0.40,             # 개선 속도 둔화 (가장 직접적)
        w_plateau: float = 0.25,           # val_acc plateau
        w_gap: float = 0.20,               # generalization gap 증가
        w_grad: float = 0.15,              # grad norm 불안정
 
        # ── 전환 트리거 임계값 ─────────────────────────────────────────
        score_threshold: float = 0.5,      # 이 점수 이상이면 전환
    ):
        """
        개선된 DynamicSwitcher:
 
        [필요조건] — AND로 결합, 모두 만족해야 충분조건 평가 진입
          1. epoch >= min_switch_epoch
          2. train loss가 안정 구간에 진입 (std < threshold)
 
        [충분조건] — 가중합 점수로 평가, score >= score_threshold 면 전환
          1. val_acc 개선 속도 둔화 (slope) ← 가장 직접적인 신호
          2. val_acc plateau (기존 신호 개선)
          3. generalization gap 증가 (기존)
          4. grad norm 불안정 (기존)
 
        기존 구조 대비 개선점:
          - 필요조건/충분조건 분리로 오탐(false positive) 감소
          - binary OR → 가중합 점수로 신호 노이즈에 강건
          - 개선 속도(slope) 신호 추가: plateau보다 조기·정확 감지
          - loss 안정성 체크: 불안정한 상태에서의 전환 방지
        """
        # 필요조건
        self.min_switch_epoch = min_switch_epoch
        self.loss_stable_window = loss_stable_window
        self.loss_stable_std_threshold = loss_stable_std_threshold
 
        # 충분조건: 개선 속도
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold
 
        # 충분조건: 기존 신호
        self.beta_ema = beta_ema
        self.history_window = history_window
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self.gap_threshold = gap_increase_threshold
        self.grad_norm_threshold = grad_norm_increase_threshold
        self.oscillation_threshold = oscillation_threshold
 
        # 가중치
        self.w_slope  = w_slope
        self.w_plateau = w_plateau
        self.w_gap    = w_gap
        self.w_grad   = w_grad
        self.score_threshold = score_threshold
 
        # 내부 상태
        self.val_acc_history = deque(maxlen=max(slope_window + 1, history_window + 1))
        self.train_loss_history = deque(maxlen=loss_stable_window)
        self.ema_values = {}
        self.grad_norm_diff_history = deque(maxlen=history_window)
        self.ema_history = deque(maxlen=history_window + 1)
 
        self.patience_counter = 0
        self.best_val_acc = 0.0
 
    # ── EMA 업데이트 ────────────────────────────────────────────────────
    def _update_ema(self, key: str, value: float):
        if key not in self.ema_values:
            self.ema_values[key] = value
        else:
            self.ema_values[key] = (
                self.beta_ema * self.ema_values[key]
                + (1 - self.beta_ema) * value
            )
 
    # ── 필요조건 1: epoch 최소 기준 ────────────────────────────────────
    def _check_min_epoch(self, epoch: int) -> bool:
        return epoch >= self.min_switch_epoch
 
    # ── 필요조건 2: train loss 안정성 ──────────────────────────────────
    def _check_loss_stable(self) -> tuple[bool, float]:
        if len(self.train_loss_history) < self.loss_stable_window:
            return False, float('inf')
        loss_std = float(np.std(list(self.train_loss_history)))
        return loss_std < self.loss_stable_std_threshold, loss_std
 
    # ── 충분조건 1: val_acc 개선 속도 (slope) ──────────────────────────
    def _check_slope(self) -> tuple[float, float]:
        """
        최근 slope_window epoch 동안의 val_acc 선형 기울기를 계산.
        반환: (정규화된 점수 0~1, 실제 slope 값)
        """
        if len(self.val_acc_history) < self.slope_window + 1:
            return 0.0, float('inf')
 
        history_list = list(self.val_acc_history)
        recent = history_list[-self.slope_window - 1:]
        x = np.arange(len(recent))
        # 최소자승법으로 기울기 계산
        slope = float(np.polyfit(x, recent, 1)[0])
 
        # slope가 threshold 이하면 점수 부여
        # slope가 0에 가까울수록 점수 1에 가까워지도록 정규화
        if slope >= self.slope_threshold:
            score = 0.0
        else:
            # threshold~0 사이: 선형 스케일, 0 이하(감소)면 1.0
            score = min(1.0, (self.slope_threshold - slope) / self.slope_threshold)
 
        return score, slope
 
    # ── 충분조건 2: plateau ─────────────────────────────────────────────
    def _check_plateau(self) -> tuple[float, int]:
        """patience_counter를 plateau_patience로 나눈 비율을 점수로."""
        score = min(1.0, self.patience_counter / self.plateau_patience)
        return score, self.patience_counter
 
    # ── 충분조건 3: generalization gap 증가 ────────────────────────────
    def _check_gap(self) -> tuple[float, float]:
        if len(self.ema_history) <= self.history_window:
            return 0.0, 0.0
        gap_current    = self.ema_history[-1]['ema_gap']
        gap_historical = self.ema_history[0]['ema_gap']
        gap_delta = gap_current - gap_historical
 
        if gap_delta <= 0:
            score = 0.0
        else:
            # gap_threshold의 2배를 최대로 정규화
            score = min(1.0, gap_delta / (self.gap_threshold * 2))
        return score, gap_delta
 
    # ── 충분조건 4: grad norm 불안정 ───────────────────────────────────
    def _check_grad_norm(self) -> tuple[float, float]:
        if len(self.ema_history) <= self.history_window:
            return 0.0, 0.0
        grad_current    = self.ema_history[-1]['ema_grad_norm']
        grad_historical = self.ema_history[0]['ema_grad_norm']
 
        if grad_historical < 1e-6:
            grad_historical = 1e-6
        relative_increase = (grad_current / grad_historical) - 1.0
 
        # oscillation 체크도 통합
        sign_changes = 0
        if len(self.grad_norm_diff_history) >= self.history_window:
            diffs = list(self.grad_norm_diff_history)
            for i in range(len(diffs) - 1):
                if diffs[i] != diffs[i + 1]:
                    sign_changes += 1
 
        is_oscillating = sign_changes >= self.oscillation_threshold
        is_increasing  = relative_increase > self.grad_norm_threshold
 
        if is_oscillating and is_increasing:
            score = 1.0
        elif is_oscillating or is_increasing:
            score = 0.6
        else:
            score = 0.0
 
        return score, relative_increase
 
    # ── 메인 step ───────────────────────────────────────────────────────
    def step(
        self,
        epoch: int,
        train_acc: float,
        val_acc: float,
        train_loss: float,
        grad_norm: float,
    ) -> bool:
        """
        매 epoch 호출. 전환 여부를 bool로 반환.
 
        Args:
            epoch      : 현재 epoch (0-indexed)
            train_acc  : 훈련 정확도 (0~1)
            val_acc    : 검증 정확도 (0~1)
            train_loss : 훈련 손실 (안정성 판단용)
            grad_norm  : 평균 gradient norm
        """
        # 히스토리 업데이트
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
 
        # grad norm diff 히스토리
        if len(self.ema_history) > 1:
            diff = self.ema_history[-1]['ema_grad_norm'] - self.ema_history[-2]['ema_grad_norm']
            if diff != 0:
                self.grad_norm_diff_history.append(np.sign(diff))
 
        # plateau 카운터 갱신
        if val_acc > self.best_val_acc + self.plateau_min_delta:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1
 
        # ── 필요조건 평가 ─────────────────────────────────────────────
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
 
        # ── 충분조건 점수 계산 ────────────────────────────────────────
        s_slope,   slope      = self._check_slope()
        s_plateau, p_count    = self._check_plateau()
        s_gap,     gap_delta  = self._check_gap()
        s_grad,    grad_rel   = self._check_grad_norm()
 
        score = (
            self.w_slope   * s_slope
            + self.w_plateau * s_plateau
            + self.w_gap     * s_gap
            + self.w_grad    * s_grad
        )
 
        print(
            f"[Switcher] Score: {score:.3f}/{self.score_threshold:.2f} | "
            f"slope={slope:.4f}%/ep(s={s_slope:.2f}) | "
            f"plateau={p_count}/{self.plateau_patience}(s={s_plateau:.2f}) | "
            f"gap_Δ={gap_delta:.3f}(s={s_gap:.2f}) | "
            f"grad_rel={grad_rel:.3f}(s={s_grad:.2f})"
        )
 
        if score >= self.score_threshold:
            print(f"[Switcher] Threshold reached! Triggering switch.")
            return True
 
        return False
    
    
class DynamicSwitcher_ver03:
    def __init__(
        self,
        min_switch_epoch: int = 150,
        slope_window: int = 30,
        slope_threshold: float = 0.01,
    ):
        """
        단순화된 DynamicSwitcher.
 
        전환 조건:
            epoch >= min_switch_epoch
            AND
            val_acc의 최근 slope < slope_threshold (%/epoch)
 
        Args:
            min_switch_epoch  : 전환을 시도하는 최소 epoch.
                                AdamW가 충분히 수렴하기 전 전환을 방지.
                                권장: 전체 epoch의 40~60%.
 
            slope_window      : val_acc 개선 속도를 측정할 구간 (epoch 수).
                                너무 짧으면 noise에 민감, 너무 길면 반응이 느림.
                                권장: 20~40.
 
            slope_threshold   : 이 값(%/epoch) 이하로 개선 속도가 떨어지면 전환.
                                0에 가까울수록 AdamW가 완전히 정체했을 때만 전환.
                                권장: 0.005~0.02.
        """
        self.min_switch_epoch = min_switch_epoch
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold
 
        self.val_acc_history = deque(maxlen=slope_window + 1)
 
    def step(self, epoch: int, val_acc: float) -> bool:
        """
        매 epoch 호출. 전환 여부를 bool로 반환.
 
        Args:
            epoch   : 현재 epoch (0-indexed)
            val_acc : 검증 정확도 (0~100 또는 0~1, 일관되게 사용)
        """
        self.val_acc_history.append(val_acc)
 
        # 필요조건: 최소 epoch 미달
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False
 
        # slope 계산에 필요한 데이터 부족
        if len(self.val_acc_history) < self.slope_window + 1:
            print(f"[Switcher] Collecting slope data... "
                  f"({len(self.val_acc_history)}/{self.slope_window + 1})")
            return False
 
        # val_acc slope 계산 (최소자승 선형 기울기)
        history = list(self.val_acc_history)
        x = np.arange(len(history))
        slope = float(np.polyfit(x, history, 1)[0])
 
        print(f"[Switcher] slope={slope:.4f} %/ep (threshold={self.slope_threshold})")
 
        if slope < self.slope_threshold:
            print(f"[Switcher] Slope below threshold → Switch triggered!")
            return True
 
        return False

class DynamicSwitcher_ver04:
    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 10,
        probe_ratio: float = 0.1,
        sim_steps: int = 10,
        gain_threshold: float = 0.2,
        rho: float = 0.05,
        weight_decay: float = 0.05,
        # ── LR restart 관련 ──────────────────────────────────────────
        initial_lr: float = 0.001,
        lr_restart_factor: float = 0.3,
    ):
        """
        네스테로프 방식의 DynamicSwitcher (LR restart 개선 버전).
 
        핵심 개선:
            전환 시 AdamW의 decay된 LR을 그대로 쓰지 않고,
            initial_lr * lr_restart_factor로 재시작.
 
            시뮬레이션도 restart_lr 기준으로 수행하여
            "실제 전환 후 상황"과 일관성 유지.
 
        LR restart가 필요한 이유:
            cosine decay로 LR이 매우 낮아진 상태에서 SAM을 적용하면
            perturbation(rho) 대비 실제 이동량이 너무 작아
            flat minima 탐색이 제대로 이뤄지지 않음.
            적당한 LR restart로 SAM이 의미 있는 탐색을 할 수 있게 함.
 
        Args:
            min_switch_epoch  : 시뮬레이션 시작 최소 epoch
            check_every       : 몇 epoch마다 시뮬레이션할지
            probe_ratio       : val set 중 사용할 비율
            sim_steps         : 시뮬레이션 step 수 (loss spike 구간 이상)
            gain_threshold    : SAM 예측 이득(%) 이 값 이상이면 전환
            rho               : SAM perturbation 크기
            weight_decay      : optimizer weight decay
            initial_lr        : 학습 초기 LR (config의 initial_lr과 동일)
            lr_restart_factor : 전환 시 LR = initial_lr * lr_restart_factor
                                너무 크면 학습 불안정, 너무 작으면 restart 효과 없음
                                권장: 0.1 ~ 0.5
        """
        self.min_switch_epoch  = min_switch_epoch
        self.check_every       = check_every
        self.probe_ratio       = probe_ratio
        self.sim_steps         = sim_steps
        self.gain_threshold    = gain_threshold
        self.rho               = rho
        self.weight_decay      = weight_decay
        self.initial_lr        = initial_lr
        self.lr_restart_factor = lr_restart_factor
 
        # 전환 시 실제로 사용할 LR (시뮬레이션과 동일한 값)
        self.restart_lr = initial_lr * lr_restart_factor
 
    # ── probe DataLoader 생성 ──────────────────────────────────────────
    def _make_probe_loader(self, val_loader: DataLoader) -> DataLoader:
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset  = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )
 
    # ── probe 정확도 측정 ─────────────────────────────────────────────
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
            total   += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0
 
    # ── SAM 시뮬레이션 (restart_lr 사용) ──────────────────────────────
    def _simulate_sam(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        adamw_state: dict,
    ) -> float:
        """
        restart_lr로 SAM(AdamW base) sim_steps 시뮬레이션.
        실제 전환 후 동작 조건과 동일하게 맞춤.
        """
        from common.optimizer import SAM
 
        sim_sam = SAM(
            model.parameters(),
            optim.AdamW,
            rho=self.rho,
            lr=self.restart_lr,      # decay된 LR 아닌 restart_lr 사용
            weight_decay=self.weight_decay,
        )
 
        # AdamW momentum state 이전 (실제 전환과 동일 조건)
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.first_step(zero_grad=True)
 
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.second_step(zero_grad=True)
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── AdamW 시뮬레이션 (현재 decay된 LR 유지) ───────────────────────
    def _simulate_adamw(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        current_lr: float,
        adamw_state: dict,
    ) -> float:
        """
        AdamW는 현재 decay된 LR 그대로 sim_steps 시뮬레이션.
        "전환하지 않고 그대로 가면 어떻게 되는가"를 반영.
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_adamw.step()
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── 메인 step ───────────────────────────────────────────────────────
    def step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> bool:
        # ── 필요조건: 최소 epoch ────────────────────────────────────────
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False
 
        # ── check_every 주기 체크 ───────────────────────────────────────
        elapsed = epoch - self.min_switch_epoch
        if elapsed > 0 and elapsed % self.check_every != 0:
            remaining = self.check_every - (elapsed % self.check_every)
            print(f"[Switcher] Skipping simulation (next check in {remaining} epoch)")
            return False
 
        # ── 시뮬레이션 준비 ─────────────────────────────────────────────
        current_lr  = optimizer.param_groups[0]['lr']
        adamw_state = deepcopy(optimizer.state_dict()['state'])
        w_saved     = deepcopy(model.state_dict())
        probe_loader = self._make_probe_loader(val_loader)
 
        print(f"\n[Switcher] Running simulation at epoch {epoch + 1}")
        print(f"  current_lr={current_lr:.6f}  restart_lr={self.restart_lr:.6f}"
              f"  rho={self.rho}  sim_steps={self.sim_steps}"
              f"  probe={self.probe_ratio*100:.0f}% of val")
 
        # ── SAM 시뮬레이션 (restart_lr 사용) ────────────────────────────
        acc_sam = self._simulate_sam(
            model, probe_loader, criterion, device, adamw_state
        )
        model.load_state_dict(w_saved)
 
        # ── AdamW 시뮬레이션 (현재 LR 유지) ─────────────────────────────
        acc_adamw = self._simulate_adamw(
            model, probe_loader, criterion, device, current_lr, adamw_state
        )
        model.load_state_dict(w_saved)
 
        # ── 결과 판단 ───────────────────────────────────────────────────
        predicted_gain = acc_sam - acc_adamw
        print(f"  SAM(restart)={acc_sam:.2f}%  AdamW(current)={acc_adamw:.2f}%"
              f"  gain={predicted_gain:+.2f}%  (threshold={self.gain_threshold:+.2f}%)")
 
        if predicted_gain >= self.gain_threshold:
            print(f"  → Gain exceeds threshold. Switch triggered!\n")
            return True
 
        print(f"  → Gain insufficient. Keep AdamW\n")
        return False
    
    
class DynamicSwitcher_ver05:
    """
    ver04 대비 핵심 변경 2가지:
 
    [변경 1] 공정한 시뮬레이션 비교
        ver04: SAM(restart_lr) vs AdamW(current_lr)
          → AdamW의 LR이 cosine decay로 고갈되면
            SAM이 실제로 나아서가 아니라 AdamW가 못 움직여서 전환됨.
        ver05: SAM(restart_lr) vs AdamW(restart_lr)
          → 동일 LR 조건에서 비교하므로, 순수하게
            "SAM perturbation이 현재 가중치 상태에서 이득인가?"를 측정.
 
    [변경 2] 전환 후 rho 분리 (switch_rho)
        ver04: rho 하나로 시뮬레이션과 전환 후 SAM 모두 사용
        ver05: switch_rho를 별도 도입
          → 시뮬레이션과 실제 전환 후 SAM 모두 switch_rho 사용
          → 이미 수렴한 가중치에 rho=0.2는 과도 → 0.05로 축소
    """
 
    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 10,
        probe_ratio: float = 0.1,
        sim_steps: int = 10,
        gain_threshold: float = 0.2,
        # ── 전환 후 SAM 전용 rho (ver05 핵심 변경) ────────────
        switch_rho: float = 0.05,
        weight_decay: float = 0.05,
        # ── LR restart 관련 ───────────────────────────────────
        initial_lr: float = 0.001,
        lr_restart_factor: float = 0.3,
    ):
        """
        Args:
            min_switch_epoch  : 시뮬레이션 시작 최소 epoch
            check_every       : 몇 epoch마다 시뮬레이션할지
            probe_ratio       : val set 중 사용할 비율
            sim_steps         : 시뮬레이션 step 수 (loss spike 구간 이상)
            gain_threshold    : SAM 예측 이득(%) 이 값 이상이면 전환
            switch_rho        : [NEW] 전환 후 SAM에서 사용할 perturbation 크기
                                수렴된 가중치에는 작은 rho가 적절 (권장: 0.03~0.10)
            weight_decay      : optimizer weight decay
            initial_lr        : 학습 초기 LR
            lr_restart_factor : 전환 시 LR = initial_lr * lr_restart_factor
        """
        self.min_switch_epoch  = min_switch_epoch
        self.check_every       = check_every
        self.probe_ratio       = probe_ratio
        self.sim_steps         = sim_steps
        self.gain_threshold    = gain_threshold
        self.switch_rho        = switch_rho
        self.weight_decay      = weight_decay
        self.initial_lr        = initial_lr
        self.lr_restart_factor = lr_restart_factor
        self.restart_lr = initial_lr * lr_restart_factor
 
    # ── probe DataLoader 생성 ──────────────────────────────────
    def _make_probe_loader(self, val_loader: DataLoader) -> DataLoader:
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset  = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )
 
    # ── probe 정확도 측정 ──────────────────────────────────────
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
            total   += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0
 
    # ── SAM 시뮬레이션 (restart_lr + switch_rho) ──────────────
    def _simulate_sam(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        adamw_state: dict,
    ) -> float:
        from common.optimizer import SAM
 
        sim_sam = SAM(
            model.parameters(),
            optim.AdamW,
            rho=self.switch_rho,       # [변경] rho → switch_rho
            lr=self.restart_lr,
            weight_decay=self.weight_decay,
        )
 
        # AdamW momentum state 이전
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.first_step(zero_grad=True)
 
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.second_step(zero_grad=True)
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── AdamW 시뮬레이션 (★ ver05 핵심 변경: restart_lr 사용) ─
    def _simulate_adamw(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        adamw_state: dict,
    ) -> float:
        """
        [ver05 핵심 변경]
        ver04: current_lr(cosine decay된 값) 사용
          → LR 고갈 시 AdamW가 못 움직여서 불공정
        ver05: restart_lr 사용 (SAM과 동일 조건)
          → 순수하게 "같은 LR에서 SAM perturbation이 이득인가?" 측정
        """
        sim_adamw = optim.AdamW(
            model.parameters(),
            lr=self.restart_lr,            # [핵심 변경] current_lr → restart_lr
            weight_decay=self.weight_decay,
        )
 
        # momentum state 이전 (SAM 시뮬레이션과 동일 조건)
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_adamw.step()
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── 메인 step ──────────────────────────────────────────────
    def step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> bool:
        # ── 필요조건: 최소 epoch ─────────────────────────────
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False
 
        # ── check_every 주기 체크 ────────────────────────────
        elapsed = epoch - self.min_switch_epoch
        if elapsed > 0 and elapsed % self.check_every != 0:
            remaining = self.check_every - (elapsed % self.check_every)
            print(f"[Switcher] Skipping simulation (next check in {remaining} epoch)")
            return False
 
        # ── 시뮬레이션 준비 ──────────────────────────────────
        current_lr  = optimizer.param_groups[0]['lr']
        adamw_state = deepcopy(optimizer.state_dict()['state'])
        w_saved     = deepcopy(model.state_dict())
        probe_loader = self._make_probe_loader(val_loader)
 
        print(f"\n[Switcher] Running simulation at epoch {epoch + 1}")
        print(f"  current_lr={current_lr:.6f}  restart_lr={self.restart_lr:.6f}"
              f"  switch_rho={self.switch_rho}  sim_steps={self.sim_steps}"
              f"  probe={self.probe_ratio*100:.0f}% of val")
        # [ver05] 양쪽 모두 restart_lr 사용함을 명시
        print(f"  [Fair comparison] Both SAM and AdamW simulated at restart_lr={self.restart_lr:.6f}")
 
        # ── SAM 시뮬레이션 (restart_lr + switch_rho) ─────────
        acc_sam = self._simulate_sam(
            model, probe_loader, criterion, device, adamw_state
        )
        model.load_state_dict(w_saved)
 
        # ── AdamW 시뮬레이션 (★ restart_lr 사용) ─────────────
        acc_adamw = self._simulate_adamw(
            model, probe_loader, criterion, device, adamw_state
        )
        model.load_state_dict(w_saved)
 
        # ── 결과 판단 ────────────────────────────────────────
        predicted_gain = acc_sam - acc_adamw
        print(f"  SAM(restart)={acc_sam:.2f}%  AdamW(restart)={acc_adamw:.2f}%"
              f"  gain={predicted_gain:+.2f}%  (threshold={self.gain_threshold:+.2f}%)")
 
        if predicted_gain >= self.gain_threshold:
            print(f"  → Gain exceeds threshold. Switch triggered!\n")
            return True
 
        print(f"  → Gain insufficient. Keep AdamW\n")
        return False
    
class DynamicSwitcher_ver06:
    """
    ver05 대비 핵심 변경 2가지:
 
    [변경 1] 시뮬레이션 신뢰성 강화
        ver05: probe_ratio=0.1, sim_steps=10, gain_threshold=0.2
          → 10% probe set + 10 step에서 accuracy 측정 분산 ~1-2%
          → gain=+0.40%로 전환 트리거됐지만 노이즈 범위 안의 값
        ver06: probe_ratio=0.3, sim_steps=20, gain_threshold=1.0
          → 30% probe set(~3000 샘플) + 20 step으로 분산 대폭 감소
          → threshold 1.0%로 높여 노이즈에 의한 오전환 방지
 
    [변경 2] rho_max로 시뮬레이션 (rho warmup과 연동)
        ver05: switch_rho=0.05 고정 → SAM ≈ 느린 AdamW
        ver06: 시뮬레이션에 rho_max 사용 (warmup 후 최종 목표 rho)
          → "rho_max에서 SAM이 실제로 이득인가?"를 직접 측정
          → rho warmup 자체는 main script에서 처리
               (전환 후 rho_min → rho_max 점진 증가)
    """
 
    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 10,
        # ── 시뮬레이션 강화 (ver06 핵심 변경 1) ──────────────
        probe_ratio: float = 0.3,
        sim_steps: int = 20,
        gain_threshold: float = 1.0,
        # ── rho 설정 ─────────────────────────────────────────
        rho_max: float = 0.15,
        weight_decay: float = 0.05,
        # ── LR restart ───────────────────────────────────────
        initial_lr: float = 0.001,
        lr_restart_factor: float = 0.3,
    ):
        """
        Args:
            min_switch_epoch  : 시뮬레이션 시작 최소 epoch
            check_every       : 몇 epoch마다 시뮬레이션할지
            probe_ratio       : [강화] val set 중 사용할 비율 (0.1 → 0.3)
            sim_steps         : [강화] 시뮬레이션 step 수 (10 → 20)
            gain_threshold    : [강화] 전환 기준 (0.2% → 1.0%)
            rho_max           : [변경] SAM 시뮬레이션 및 warmup 후 목표 rho
            weight_decay      : optimizer weight decay
            initial_lr        : 학습 초기 LR
            lr_restart_factor : 전환 시 LR = initial_lr * lr_restart_factor
        """
        self.min_switch_epoch  = min_switch_epoch
        self.check_every       = check_every
        self.probe_ratio       = probe_ratio
        self.sim_steps         = sim_steps
        self.gain_threshold    = gain_threshold
        self.rho_max           = rho_max
        self.weight_decay      = weight_decay
        self.initial_lr        = initial_lr
        self.lr_restart_factor = lr_restart_factor
        self.restart_lr = initial_lr * lr_restart_factor
 
    # ── probe DataLoader ───────────────────────────────────────
    def _make_probe_loader(self, val_loader: DataLoader) -> DataLoader:
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset  = Subset(dataset, indices)
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
            total   += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0
 
    # ── SAM 시뮬레이션 (rho_max 사용) ─────────────────────────
    def _simulate_sam(
        self, model: nn.Module, probe_loader: DataLoader,
        criterion: nn.Module, device: torch.device,
        adamw_state: dict,
    ) -> float:
        from common.optimizer import SAM
 
        sim_sam = SAM(
            model.parameters(), optim.AdamW,
            rho=self.rho_max,          # warmup 후 최종 목표 rho로 시뮬레이션
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.first_step(zero_grad=True)
 
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.second_step(zero_grad=True)
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── AdamW 시뮬레이션 (restart_lr, 공정 비교 유지) ─────────
    def _simulate_adamw(
        self, model: nn.Module, probe_loader: DataLoader,
        criterion: nn.Module, device: torch.device,
        adamw_state: dict,
    ) -> float:
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_adamw.step()
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── 메인 step ──────────────────────────────────────────────
    def step(
        self, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
        val_loader: DataLoader, criterion: nn.Module, device: torch.device,
    ) -> bool:
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False
 
        elapsed = epoch - self.min_switch_epoch
        if elapsed > 0 and elapsed % self.check_every != 0:
            remaining = self.check_every - (elapsed % self.check_every)
            print(f"[Switcher] Skipping simulation (next check in {remaining} epoch)")
            return False
 
        current_lr  = optimizer.param_groups[0]['lr']
        adamw_state = deepcopy(optimizer.state_dict()['state'])
        w_saved     = deepcopy(model.state_dict())
        probe_loader = self._make_probe_loader(val_loader)
 
        print(f"\n[Switcher] Running simulation at epoch {epoch + 1}")
        print(f"  current_lr={current_lr:.6f}  restart_lr={self.restart_lr:.6f}"
              f"  rho_max={self.rho_max}  sim_steps={self.sim_steps}"
              f"  probe={self.probe_ratio*100:.0f}% of val")
        print(f"  [Fair comparison] Both at restart_lr={self.restart_lr:.6f},"
              f" SAM uses rho_max={self.rho_max}")
 
        acc_sam = self._simulate_sam(
            model, probe_loader, criterion, device, adamw_state)
        model.load_state_dict(w_saved)
 
        acc_adamw = self._simulate_adamw(
            model, probe_loader, criterion, device, adamw_state)
        model.load_state_dict(w_saved)
 
        predicted_gain = acc_sam - acc_adamw
        print(f"  SAM(rho={self.rho_max})={acc_sam:.2f}%"
              f"  AdamW={acc_adamw:.2f}%"
              f"  gain={predicted_gain:+.2f}%  (threshold={self.gain_threshold:+.2f}%)")
 
        if predicted_gain >= self.gain_threshold:
            print(f"  → Gain exceeds threshold. Switch triggered!\n")
            return True
 
        print(f"  → Gain insufficient. Keep AdamW\n")
        return False
    

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
    
    
@dataclass
class DynamicSwitcher_ver02:
    def __init__(
        self,
        # ── 필요조건 파라미터 ──────────────────────────────────────────
        min_switch_epoch: int = 150,
        loss_stable_window: int = 10,       # loss 안정성 판단 윈도우
        loss_stable_std_threshold: float = 0.02,  # loss std 이 이하면 안정
 
        # ── 충분조건: 개선 속도 ────────────────────────────────────────
        slope_window: int = 30,             # 개선 속도 측정 윈도우 (epoch)
        slope_threshold: float = 0.01,      # %/epoch 이하면 "개선 둔화"
 
        # ── 충분조건: 기존 신호 (보조) ────────────────────────────────
        beta_ema: float = 0.9,
        history_window: int = 20,
        plateau_patience: int = 20,
        plateau_min_delta: float = 0.005,
        gap_increase_threshold: float = 0.02,
        grad_norm_increase_threshold: float = 0.1,
        oscillation_threshold: int = 5,
 
        # ── 점수 가중치 ────────────────────────────────────────────────
        # 각 충분조건 신호의 기여 가중치 (합이 1이 되도록 설계)
        w_slope: float = 0.40,             # 개선 속도 둔화 (가장 직접적)
        w_plateau: float = 0.25,           # val_acc plateau
        w_gap: float = 0.20,               # generalization gap 증가
        w_grad: float = 0.15,              # grad norm 불안정
 
        # ── 전환 트리거 임계값 ─────────────────────────────────────────
        score_threshold: float = 0.5,      # 이 점수 이상이면 전환
    ):
        """
        개선된 DynamicSwitcher:
 
        [필요조건] — AND로 결합, 모두 만족해야 충분조건 평가 진입
          1. epoch >= min_switch_epoch
          2. train loss가 안정 구간에 진입 (std < threshold)
 
        [충분조건] — 가중합 점수로 평가, score >= score_threshold 면 전환
          1. val_acc 개선 속도 둔화 (slope) ← 가장 직접적인 신호
          2. val_acc plateau (기존 신호 개선)
          3. generalization gap 증가 (기존)
          4. grad norm 불안정 (기존)
 
        기존 구조 대비 개선점:
          - 필요조건/충분조건 분리로 오탐(false positive) 감소
          - binary OR → 가중합 점수로 신호 노이즈에 강건
          - 개선 속도(slope) 신호 추가: plateau보다 조기·정확 감지
          - loss 안정성 체크: 불안정한 상태에서의 전환 방지
        """
        # 필요조건
        self.min_switch_epoch = min_switch_epoch
        self.loss_stable_window = loss_stable_window
        self.loss_stable_std_threshold = loss_stable_std_threshold
 
        # 충분조건: 개선 속도
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold
 
        # 충분조건: 기존 신호
        self.beta_ema = beta_ema
        self.history_window = history_window
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self.gap_threshold = gap_increase_threshold
        self.grad_norm_threshold = grad_norm_increase_threshold
        self.oscillation_threshold = oscillation_threshold
 
        # 가중치
        self.w_slope  = w_slope
        self.w_plateau = w_plateau
        self.w_gap    = w_gap
        self.w_grad   = w_grad
        self.score_threshold = score_threshold
 
        # 내부 상태
        self.val_acc_history = deque(maxlen=max(slope_window + 1, history_window + 1))
        self.train_loss_history = deque(maxlen=loss_stable_window)
        self.ema_values = {}
        self.grad_norm_diff_history = deque(maxlen=history_window)
        self.ema_history = deque(maxlen=history_window + 1)
 
        self.patience_counter = 0
        self.best_val_acc = 0.0
 
    # ── EMA 업데이트 ────────────────────────────────────────────────────
    def _update_ema(self, key: str, value: float):
        if key not in self.ema_values:
            self.ema_values[key] = value
        else:
            self.ema_values[key] = (
                self.beta_ema * self.ema_values[key]
                + (1 - self.beta_ema) * value
            )
 
    # ── 필요조건 1: epoch 최소 기준 ────────────────────────────────────
    def _check_min_epoch(self, epoch: int) -> bool:
        return epoch >= self.min_switch_epoch
 
    # ── 필요조건 2: train loss 안정성 ──────────────────────────────────
    def _check_loss_stable(self) -> tuple[bool, float]:
        if len(self.train_loss_history) < self.loss_stable_window:
            return False, float('inf')
        loss_std = float(np.std(list(self.train_loss_history)))
        return loss_std < self.loss_stable_std_threshold, loss_std
 
    # ── 충분조건 1: val_acc 개선 속도 (slope) ──────────────────────────
    def _check_slope(self) -> tuple[float, float]:
        """
        최근 slope_window epoch 동안의 val_acc 선형 기울기를 계산.
        반환: (정규화된 점수 0~1, 실제 slope 값)
        """
        if len(self.val_acc_history) < self.slope_window + 1:
            return 0.0, float('inf')
 
        history_list = list(self.val_acc_history)
        recent = history_list[-self.slope_window - 1:]
        x = np.arange(len(recent))
        # 최소자승법으로 기울기 계산
        slope = float(np.polyfit(x, recent, 1)[0])
 
        # slope가 threshold 이하면 점수 부여
        # slope가 0에 가까울수록 점수 1에 가까워지도록 정규화
        if slope >= self.slope_threshold:
            score = 0.0
        else:
            # threshold~0 사이: 선형 스케일, 0 이하(감소)면 1.0
            score = min(1.0, (self.slope_threshold - slope) / self.slope_threshold)
 
        return score, slope
 
    # ── 충분조건 2: plateau ─────────────────────────────────────────────
    def _check_plateau(self) -> tuple[float, int]:
        """patience_counter를 plateau_patience로 나눈 비율을 점수로."""
        score = min(1.0, self.patience_counter / self.plateau_patience)
        return score, self.patience_counter
 
    # ── 충분조건 3: generalization gap 증가 ────────────────────────────
    def _check_gap(self) -> tuple[float, float]:
        if len(self.ema_history) <= self.history_window:
            return 0.0, 0.0
        gap_current    = self.ema_history[-1]['ema_gap']
        gap_historical = self.ema_history[0]['ema_gap']
        gap_delta = gap_current - gap_historical
 
        if gap_delta <= 0:
            score = 0.0
        else:
            # gap_threshold의 2배를 최대로 정규화
            score = min(1.0, gap_delta / (self.gap_threshold * 2))
        return score, gap_delta
 
    # ── 충분조건 4: grad norm 불안정 ───────────────────────────────────
    def _check_grad_norm(self) -> tuple[float, float]:
        if len(self.ema_history) <= self.history_window:
            return 0.0, 0.0
        grad_current    = self.ema_history[-1]['ema_grad_norm']
        grad_historical = self.ema_history[0]['ema_grad_norm']
 
        if grad_historical < 1e-6:
            grad_historical = 1e-6
        relative_increase = (grad_current / grad_historical) - 1.0
 
        # oscillation 체크도 통합
        sign_changes = 0
        if len(self.grad_norm_diff_history) >= self.history_window:
            diffs = list(self.grad_norm_diff_history)
            for i in range(len(diffs) - 1):
                if diffs[i] != diffs[i + 1]:
                    sign_changes += 1
 
        is_oscillating = sign_changes >= self.oscillation_threshold
        is_increasing  = relative_increase > self.grad_norm_threshold
 
        if is_oscillating and is_increasing:
            score = 1.0
        elif is_oscillating or is_increasing:
            score = 0.6
        else:
            score = 0.0
 
        return score, relative_increase
 
    # ── 메인 step ───────────────────────────────────────────────────────
    def step(
        self,
        epoch: int,
        train_acc: float,
        val_acc: float,
        train_loss: float,
        grad_norm: float,
    ) -> bool:
        """
        매 epoch 호출. 전환 여부를 bool로 반환.
 
        Args:
            epoch      : 현재 epoch (0-indexed)
            train_acc  : 훈련 정확도 (0~1)
            val_acc    : 검증 정확도 (0~1)
            train_loss : 훈련 손실 (안정성 판단용)
            grad_norm  : 평균 gradient norm
        """
        # 히스토리 업데이트
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
 
        # grad norm diff 히스토리
        if len(self.ema_history) > 1:
            diff = self.ema_history[-1]['ema_grad_norm'] - self.ema_history[-2]['ema_grad_norm']
            if diff != 0:
                self.grad_norm_diff_history.append(np.sign(diff))
 
        # plateau 카운터 갱신
        if val_acc > self.best_val_acc + self.plateau_min_delta:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1
 
        # ── 필요조건 평가 ─────────────────────────────────────────────
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
 
        # ── 충분조건 점수 계산 ────────────────────────────────────────
        s_slope,   slope      = self._check_slope()
        s_plateau, p_count    = self._check_plateau()
        s_gap,     gap_delta  = self._check_gap()
        s_grad,    grad_rel   = self._check_grad_norm()
 
        score = (
            self.w_slope   * s_slope
            + self.w_plateau * s_plateau
            + self.w_gap     * s_gap
            + self.w_grad    * s_grad
        )
 
        print(
            f"[Switcher] Score: {score:.3f}/{self.score_threshold:.2f} | "
            f"slope={slope:.4f}%/ep(s={s_slope:.2f}) | "
            f"plateau={p_count}/{self.plateau_patience}(s={s_plateau:.2f}) | "
            f"gap_Δ={gap_delta:.3f}(s={s_gap:.2f}) | "
            f"grad_rel={grad_rel:.3f}(s={s_grad:.2f})"
        )
 
        if score >= self.score_threshold:
            print(f"[Switcher] Threshold reached! Triggering switch.")
            return True
 
        return False
    
    
class DynamicSwitcher_ver03:
    def __init__(
        self,
        min_switch_epoch: int = 150,
        slope_window: int = 30,
        slope_threshold: float = 0.01,
    ):
        """
        단순화된 DynamicSwitcher.
 
        전환 조건:
            epoch >= min_switch_epoch
            AND
            val_acc의 최근 slope < slope_threshold (%/epoch)
 
        Args:
            min_switch_epoch  : 전환을 시도하는 최소 epoch.
                                AdamW가 충분히 수렴하기 전 전환을 방지.
                                권장: 전체 epoch의 40~60%.
 
            slope_window      : val_acc 개선 속도를 측정할 구간 (epoch 수).
                                너무 짧으면 noise에 민감, 너무 길면 반응이 느림.
                                권장: 20~40.
 
            slope_threshold   : 이 값(%/epoch) 이하로 개선 속도가 떨어지면 전환.
                                0에 가까울수록 AdamW가 완전히 정체했을 때만 전환.
                                권장: 0.005~0.02.
        """
        self.min_switch_epoch = min_switch_epoch
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold
 
        self.val_acc_history = deque(maxlen=slope_window + 1)
 
    def step(self, epoch: int, val_acc: float) -> bool:
        """
        매 epoch 호출. 전환 여부를 bool로 반환.
 
        Args:
            epoch   : 현재 epoch (0-indexed)
            val_acc : 검증 정확도 (0~100 또는 0~1, 일관되게 사용)
        """
        self.val_acc_history.append(val_acc)
 
        # 필요조건: 최소 epoch 미달
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False
 
        # slope 계산에 필요한 데이터 부족
        if len(self.val_acc_history) < self.slope_window + 1:
            print(f"[Switcher] Collecting slope data... "
                  f"({len(self.val_acc_history)}/{self.slope_window + 1})")
            return False
 
        # val_acc slope 계산 (최소자승 선형 기울기)
        history = list(self.val_acc_history)
        x = np.arange(len(history))
        slope = float(np.polyfit(x, history, 1)[0])
 
        print(f"[Switcher] slope={slope:.4f} %/ep (threshold={self.slope_threshold})")
 
        if slope < self.slope_threshold:
            print(f"[Switcher] Slope below threshold → Switch triggered!")
            return True
 
        return False

class DynamicSwitcher_ver04:
    def __init__(
        self,
        min_switch_epoch: int = 150,
        check_every: int = 10,
        probe_ratio: float = 0.1,
        sim_steps: int = 10,
        gain_threshold: float = 0.2,
        rho: float = 0.05,
        weight_decay: float = 0.05,
        # ── LR restart 관련 ──────────────────────────────────────────
        initial_lr: float = 0.001,
        lr_restart_factor: float = 0.3,
    ):
        """
        네스테로프 방식의 DynamicSwitcher (LR restart 개선 버전).
 
        핵심 개선:
            전환 시 AdamW의 decay된 LR을 그대로 쓰지 않고,
            initial_lr * lr_restart_factor로 재시작.
 
            시뮬레이션도 restart_lr 기준으로 수행하여
            "실제 전환 후 상황"과 일관성 유지.
 
        LR restart가 필요한 이유:
            cosine decay로 LR이 매우 낮아진 상태에서 SAM을 적용하면
            perturbation(rho) 대비 실제 이동량이 너무 작아
            flat minima 탐색이 제대로 이뤄지지 않음.
            적당한 LR restart로 SAM이 의미 있는 탐색을 할 수 있게 함.
 
        Args:
            min_switch_epoch  : 시뮬레이션 시작 최소 epoch
            check_every       : 몇 epoch마다 시뮬레이션할지
            probe_ratio       : val set 중 사용할 비율
            sim_steps         : 시뮬레이션 step 수 (loss spike 구간 이상)
            gain_threshold    : SAM 예측 이득(%) 이 값 이상이면 전환
            rho               : SAM perturbation 크기
            weight_decay      : optimizer weight decay
            initial_lr        : 학습 초기 LR (config의 initial_lr과 동일)
            lr_restart_factor : 전환 시 LR = initial_lr * lr_restart_factor
                                너무 크면 학습 불안정, 너무 작으면 restart 효과 없음
                                권장: 0.1 ~ 0.5
        """
        self.min_switch_epoch  = min_switch_epoch
        self.check_every       = check_every
        self.probe_ratio       = probe_ratio
        self.sim_steps         = sim_steps
        self.gain_threshold    = gain_threshold
        self.rho               = rho
        self.weight_decay      = weight_decay
        self.initial_lr        = initial_lr
        self.lr_restart_factor = lr_restart_factor
 
        # 전환 시 실제로 사용할 LR (시뮬레이션과 동일한 값)
        self.restart_lr = initial_lr * lr_restart_factor
 
    # ── probe DataLoader 생성 ──────────────────────────────────────────
    def _make_probe_loader(self, val_loader: DataLoader) -> DataLoader:
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset  = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )
 
    # ── probe 정확도 측정 ─────────────────────────────────────────────
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
            total   += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0
 
    # ── SAM 시뮬레이션 (restart_lr 사용) ──────────────────────────────
    def _simulate_sam(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        adamw_state: dict,
    ) -> float:
        """
        restart_lr로 SAM(AdamW base) sim_steps 시뮬레이션.
        실제 전환 후 동작 조건과 동일하게 맞춤.
        """
        from common.optimizer import SAM
 
        sim_sam = SAM(
            model.parameters(),
            optim.AdamW,
            rho=self.rho,
            lr=self.restart_lr,      # decay된 LR 아닌 restart_lr 사용
            weight_decay=self.weight_decay,
        )
 
        # AdamW momentum state 이전 (실제 전환과 동일 조건)
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.first_step(zero_grad=True)
 
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_sam.second_step(zero_grad=True)
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── AdamW 시뮬레이션 (현재 decay된 LR 유지) ───────────────────────
    def _simulate_adamw(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        current_lr: float,
        adamw_state: dict,
    ) -> float:
        """
        AdamW는 현재 decay된 LR 그대로 sim_steps 시뮬레이션.
        "전환하지 않고 그대로 가면 어떻게 되는가"를 반영.
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
                loss    = criterion(outputs, targets)
            loss.backward()
            sim_adamw.step()
 
        return self._probe_accuracy(model, probe_loader, criterion, device)
 
    # ── 메인 step ───────────────────────────────────────────────────────
    def step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> bool:
        # ── 필요조건: 최소 epoch ────────────────────────────────────────
        if epoch < self.min_switch_epoch:
            print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            return False
 
        # ── check_every 주기 체크 ───────────────────────────────────────
        elapsed = epoch - self.min_switch_epoch
        if elapsed > 0 and elapsed % self.check_every != 0:
            remaining = self.check_every - (elapsed % self.check_every)
            print(f"[Switcher] Skipping simulation (next check in {remaining} epoch)")
            return False
 
        # ── 시뮬레이션 준비 ─────────────────────────────────────────────
        current_lr  = optimizer.param_groups[0]['lr']
        adamw_state = deepcopy(optimizer.state_dict()['state'])
        w_saved     = deepcopy(model.state_dict())
        probe_loader = self._make_probe_loader(val_loader)
 
        print(f"\n[Switcher] Running simulation at epoch {epoch + 1}")
        print(f"  current_lr={current_lr:.6f}  restart_lr={self.restart_lr:.6f}"
              f"  rho={self.rho}  sim_steps={self.sim_steps}"
              f"  probe={self.probe_ratio*100:.0f}% of val")
 
        # ── SAM 시뮬레이션 (restart_lr 사용) ────────────────────────────
        acc_sam = self._simulate_sam(
            model, probe_loader, criterion, device, adamw_state
        )
        model.load_state_dict(w_saved)
 
        # ── AdamW 시뮬레이션 (현재 LR 유지) ─────────────────────────────
        acc_adamw = self._simulate_adamw(
            model, probe_loader, criterion, device, current_lr, adamw_state
        )
        model.load_state_dict(w_saved)
 
        # ── 결과 판단 ───────────────────────────────────────────────────
        predicted_gain = acc_sam - acc_adamw
        print(f"  SAM(restart)={acc_sam:.2f}%  AdamW(current)={acc_adamw:.2f}%"
              f"  gain={predicted_gain:+.2f}%  (threshold={self.gain_threshold:+.2f}%)")
 
        if predicted_gain >= self.gain_threshold:
            print(f"  → Gain exceeds threshold. Switch triggered!\n")
            return True
 
        print(f"  → Gain insufficient. Keep AdamW\n")
        return False


class SharpnessAwareSwitcher:
    """
    Sharpness 기반 전환 결정기.

    패러다임 전환:
        ver04~06의 네스테로프 시뮬레이션은 "N step 후 accuracy"를 비교했지만,
        SAM의 이점은 단기 accuracy가 아니라 loss landscape의 기하학적 성질(flatness).
        단기 시뮬레이션에서 AdamW가 항상 이기는 것은 당연 (SAM은 의도적으로 느림).

        이 클래스는 질문 자체를 바꿈:
        "SAM이 AdamW보다 나은가?" (X) -> "현재 가중치가 sharp minima에 있는가?" (O)

    측정:
        sharpness = loss(w + rho * g/||g||) - loss(w)
        SAM first_step과 동일한 perturbation. Forward 2회 + Backward 1회로 측정.
        (sim_steps=20 시뮬레이션 대비 ~10배 저렴)

    전환 조건:
        epoch >= min_switch_epoch AND EMA(sharpness) >= sharpness_threshold

    첫 실험 용도:
        sharpness_threshold를 높게(999.0) 설정하면 전환 없이 sharpness만 기록.
        로그에서 sharpness 추이를 관찰한 뒤 적절한 threshold를 결정할 수 있음.
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
        Args:
            min_switch_epoch    : 전환을 고려하는 최소 epoch
            check_every         : sharpness 측정 주기 (epoch 단위)
            probe_ratio         : val set 중 측정에 사용할 비율
            sharpness_rho       : perturbation 크기 (전환 후 SAM rho_max와 동일 권장)
            sharpness_threshold : EMA(sharpness)가 이 값 이상이면 전환
            sharpness_ema_beta  : EMA 평활 계수
        """
        self.min_switch_epoch    = min_switch_epoch
        self.check_every         = check_every
        self.probe_ratio         = probe_ratio
        self.sharpness_rho       = sharpness_rho
        self.sharpness_threshold = sharpness_threshold
        self.ema_beta            = sharpness_ema_beta

        self.sharpness_history = []
        self.ema_sharpness = None

    def _make_probe_loader(self, val_loader):
        dataset = val_loader.dataset
        n_probe = max(1, int(len(dataset) * self.probe_ratio))
        indices = torch.randperm(len(dataset))[:n_probe].tolist()
        subset  = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=getattr(val_loader, 'num_workers', 0),
        )

    def _measure_sharpness(self, model, probe_loader, criterion, device):
        """
        Sharpness = loss(w + rho*g/||g||) - loss(w)

        Step 1: probe set에서 loss 계산 + gradient 축적
        Step 2: gradient 방향으로 rho만큼 perturbation
        Step 3: perturbed 가중치에서 loss 재계산
        Step 4: 원래 가중치 복원
        """
        model.train()
        model.zero_grad()

        # Step 1: original loss + gradient
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

        # Step 2: compute grad norm and perturb
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

        # Step 3: perturbed loss (no grad needed)
        total_loss_pert = 0.0
        with torch.no_grad():
            for inputs, targets in probe_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                total_loss_pert += loss.item() * inputs.size(0)

        avg_loss_pert = total_loss_pert / total_samples

        # Step 4: restore original weights
        for name, p in model.named_parameters():
            if name in old_params:
                p.data.copy_(old_params[name])
        model.zero_grad()

        return avg_loss_pert - avg_loss_orig

    def step(self, epoch, model, val_loader, criterion, device):
        """
        매 epoch 호출. 전환 여부를 bool로 반환.
        min_switch_epoch 이전에도 sharpness를 측정하여 기록.
        """
        # 측정 주기가 아니면 스킵
        if epoch % self.check_every != 0:
            if epoch < self.min_switch_epoch:
                print(f"[Switcher] Warming up... ({epoch + 1}/{self.min_switch_epoch})")
            else:
                remaining = self.check_every - (epoch % self.check_every)
                print(f"[Switcher] Next sharpness check in {remaining} epoch")
            return False

        # sharpness 측정
        probe_loader = self._make_probe_loader(val_loader)
        sharpness = self._measure_sharpness(model, probe_loader, criterion, device)

        # EMA 업데이트
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

        # 로그 (항상 출력 — threshold 설정용 데이터 수집)
        phase = "MONITOR" if epoch < self.min_switch_epoch else "EVALUATE"
        print(
            f"\n[Switcher] Sharpness at epoch {epoch + 1} [{phase}]"
            f"\n  raw={sharpness:.4f}  ema={self.ema_sharpness:.4f}"
            f"  threshold={self.sharpness_threshold}  rho={self.sharpness_rho}"
        )

        # 필요조건: 최소 epoch
        if epoch < self.min_switch_epoch:
            print(f"  -> Monitoring only ({epoch + 1}/{self.min_switch_epoch})")
            return False

        # 전환 판단
        if self.ema_sharpness >= self.sharpness_threshold:
            print(f"  -> SWITCH TRIGGERED! ema={self.ema_sharpness:.4f} >= {self.sharpness_threshold}\n")
            return True

        print(f"  -> Keep AdamW (ema={self.ema_sharpness:.4f} < {self.sharpness_threshold})\n")
        return False