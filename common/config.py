import torch


def print_config(config):
    """Pretty-print a config dictionary."""
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
        # general
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
        # general
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
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,
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

        # score weights and threshold
        'w_slope': 0.40,         # weight for slope signal (most direct)
        'w_plateau': 0.25,       # weight for plateau signal
        'w_gap': 0.20,           # weight for generalization gap signal
        'w_grad': 0.15,          # weight for grad norm signal
        'score_threshold': 0.5,  # weighted score above this triggers a switch
    }
    return config


def get_config_ver03():
    EPOCHS = 300
    WARMUP_EPOCHS = 10

    config = {
        # general
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_workers': 16,

        # dataset
        'dataset': 'CIFAR100',
        'num_classes': 100,
        'data_path': './data/',
        'image_size': 224,

        # data augmentation
        'use_randaugment': False,
        'use_autoaugment': True,
        'use_mixup': True,
        'use_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'aug_prob': 0.5,

        # model
        'model_name': 'deit_tiny_patch16_224',
        'dropout_rate': 0.0,

        # training
        'use_amp': False,
        'epochs': EPOCHS,
        'batch_size': 128,

        # AdamW
        'warmup_epochs': WARMUP_EPOCHS,
        'initial_lr': 0.001,
        'weight_decay': 0.05,

        # SAM
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,
        'rho': 0.05,

        # DynamicSwitcher
        'min_switch_epoch': 150,
        'slope_window': 30,
        'slope_threshold': 0.01,
    }
    return config


def get_config_ver04():
    EPOCHS = 300
    WARMUP_EPOCHS = 10

    config = {
        # general
        'seed': 42,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_workers': 16,

        # dataset
        'dataset': 'CIFAR100',
        'num_classes': 100,
        'data_path': './data/',
        'image_size': 224,

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
        'initial_lr': 0.001,
        'weight_decay': 0.05,

        # SAM
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,
        'rho': 0.20,

        # DynamicSwitcher
        'min_switch_epoch': 150,
        'check_every': 10,           # run simulation every N epochs
        'probe_ratio': 0.1,          # fraction of val set used in simulation
        'sim_steps': 10,             # number of simulation steps
        'gain_threshold': 0.2,       # minimum predicted SAM gain (%) to trigger switch
        'lr_restart_factor': 0.5,    # restart_lr = initial_lr * factor
    }
    return config


def get_config_ver05():
    """
    Changes from ver04:
      1. rho split into switch_rho (0.05): separate rho for post-switch SAM.
         rho=0.2 is too aggressive for already-converged weights, causing
         a sharp accuracy drop. switch_rho=0.05 provides a gentler transition.
      2. lr_restart_factor: 0.5 -> 0.3.
         In v04 logs, restart_lr=0.0005 was ~10x the current LR.
         Lowering to 0.3 reduces the transition shock (restart_lr=0.0003).
      3. DynamicSwitcher_ver05: fair simulation comparison.
         Both SAM and AdamW are simulated at restart_lr so that the measured
         gain reflects the benefit of SAM perturbation itself, not an LR advantage.
    """
    EPOCHS = 300
    WARMUP_EPOCHS = 10

    config = {
        # general
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
        'initial_lr': 0.001,
        'weight_decay': 0.05,

        # SAM
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,
        'switch_rho': 0.05,        # separate rho for post-switch SAM (ver05 change)

        # DynamicSwitcher_ver05
        'min_switch_epoch': 150,
        'check_every': 10,
        'probe_ratio': 0.1,
        'sim_steps': 10,
        'gain_threshold': 0.2,
        'lr_restart_factor': 0.3,  # reduced from 0.5 to soften transition shock
    }
    return config


def get_config_ver06():
    """
    Changes from ver05:

    1. rho warmup schedule introduced.
       Single switch_rho replaced by rho_min / rho_max / rho_warmup_epochs.
       Starts at rho_min (0.02) right after the switch and linearly increases
       to rho_max (0.15) over the warmup window. This resolves the immediate
       drop from v4 (rho=0.2 too sudden) and the ineffectiveness of v5 (rho=0.05 fixed).

    2. Stronger simulation reliability.
       probe_ratio: 0.1 -> 0.3 (reduces variance).
       sim_steps: 10 -> 20 (moves past the initial loss spike region).
       gain_threshold: 0.2 -> 1.0 (prevents spurious switches due to noise).

    3. rho_max=0.15 rationale.
       SAM_Only with rho=0.20 trained stably, so rho=0.15 is slightly conservative
       and safer for already-converged weights. Reached gradually via warmup.
    """
    EPOCHS = 300
    WARMUP_EPOCHS = 10

    config = {
        # general
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
        'initial_lr': 0.001,
        'weight_decay': 0.05,

        # SAM
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,

        # rho warmup schedule (ver06 core change)
        'rho_min': 0.02,           # starting rho right after the switch
        'rho_max': 0.15,           # target rho after warmup
        'rho_warmup_epochs': 20,   # number of epochs to linearly increase rho

        # DynamicSwitcher_ver06
        'min_switch_epoch': 150,
        'check_every': 10,
        'probe_ratio': 0.3,        # increased from 0.1
        'sim_steps': 20,           # increased from 10
        'gain_threshold': 1.0,     # increased from 0.2 to suppress noise-driven switches
        'lr_restart_factor': 0.3,
    }
    return config


def get_config_ver07():
    """
    ver07: sharpness-based switching with rho warmup.

    Lessons from ver04-06 Nesterov simulations:
        SAM cannot beat AdamW in short-horizon accuracy by design.
        The right question is not "is SAM better?" but "is the landscape sharp?".
        rho=0.05 is too small for SAM to have any effect; rho=0.2 causes an
        accuracy drop. rho warmup (0.02 -> 0.15) resolves both.

    First experiment strategy:
        Set sharpness_threshold=999.0 to monitor sharpness without switching.
        Observe the sharpness trend in the logs, then set a meaningful threshold.
        With this setting the run is effectively identical to AdamW_Only while
        collecting sharpness data.
    """
    EPOCHS = 300
    WARMUP_EPOCHS = 10

    config = {
        # general
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
        'initial_lr': 0.001,
        'weight_decay': 0.05,

        # SAM_Only strategy
        'sam_only_lr': 1e-3,
        'sam_only_rho': 0.20,

        # rho warmup applied after the switch
        'rho_min': 0.02,
        'rho_max': 0.15,
        'rho_warmup_epochs': 20,

        # SharpnessAwareSwitcher
        'min_switch_epoch': 150,
        'check_every': 5,              # sharpness measurement is cheap, so check often
        'probe_ratio': 0.2,
        'sharpness_rho': 0.15,         # same as rho_max
        'sharpness_threshold': 0.53,   # set to 999.0 for monitoring-only mode
        'sharpness_ema_beta': 0.9,

        # LR restart on switch
        'lr_restart_factor': 0.3,
    }
    return config