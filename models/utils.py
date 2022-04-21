import torch

CRITERION = {
    'regression':torch.nn.L1Loss,
}

def get_criterion(task_name):
    if task_name not in CRITERION:
        raise ValueError("Task not recognized or not supported.")
    else:
        return CRITERION[task_name]

OPTIMIZER = {
    'sgd':torch.optim.SGD,
    'adam':torch.optim.Adam,
    'adamw':torch.optim.AdamW,
}

def get_optimizer(optimizer):
    if optimizer not in OPTIMIZER:
        raise ValueError("Optimizer not recognized or not supported.")
    else:
        return OPTIMIZER[optimizer]