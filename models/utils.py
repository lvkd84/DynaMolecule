import torch

CRITERION = {

}

def get_criterion(task_name):
    if task_name not in CRITERION:
        raise ValueError("Task not recognized or not supported.")
    else:
        return CRITERION[task_name]

OPTIMIZER = {

}

def get_optimizer(optimizer):
    if optimizer not in OPTIMIZER:
        raise ValueError("Optimizer not recognized or not supported.")
    else:
        return OPTIMIZER[optimizer]