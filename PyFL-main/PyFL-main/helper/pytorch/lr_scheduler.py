import math
import torch


class CosineAnnealingLR(object):
    def __init__(self, optimizer, args, eta_min=0):
        print(args)
        self.optimizer = optimizer
        self.T_max = args["epochs"] + 0.1 - args["warm_up_epochs"]
        self.T_i = args["warm_up_epochs"]
        self.eta_min = eta_min
        self.gamma = args["gamma"]
        self.base_lr = args["lr"]
        self.init_lr = args["baseline_lr"]
        self.factor = self.base_lr / self.init_lr
        self.warm_up_epochs = args["warm_up_epochs"]
        self.epoch = 1
        if self.warm_up_epochs > 0:
            assert self.factor >= 1, "The target LR {:.3f} should be >= baseline_lr {:.2f}!".format(self.base_lr,
                                                                                                    self.init_lr)

    def step(self):
        if self.epoch < self.warm_up_epochs:
            lr = self.base_lr * 1 / self.factor * (
                    self.epoch * (self.factor - 1) / self.warm_up_epochs + 1)
        else:
            lr = self.eta_min + (self.base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * (self.epoch - self.T_i) / self.T_max)) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.epoch += 1


class MultiStepLR(object):
    def __init__(self, optimizer, args, milestones_travelled=0):
        self.optimizer = optimizer
        self.milestones_travelled = milestones_travelled
        self.milestones = args["lrmilestone"]
        self.gamma = args["gamma"]
        self.target_lr = args["lr"]
        self.init_lr = args["baseline_lr"]
        self.factor = self.target_lr / self.init_lr
        self.warm_up_epochs = args["warm_up_epochs"]
        self.epoch = 1
        if self.warm_up_epochs > 0:
            assert self.factor >= 1, "The target LR should be >= baseline_lr!"

    def step(self):
        if self.epoch < self.warm_up_epochs:
            lr = self.target_lr * 1 / self.factor * (self.epoch * (self.factor - 1) / self.warm_up_epochs + 1)
            set_current_lr(self.optimizer, lr)
        elif self.warm_up_epochs <= self.epoch < self.milestones[0]:
            set_current_lr(self.optimizer, self.target_lr)
        elif self.milestones_travelled < len(self.milestones) and self.epoch >= self.milestones[
            self.milestones_travelled]:
            print("Dampening LR at ", self.epoch)
            self.milestones_travelled += 1
            set_current_lr(self.optimizer, get_current_lr(self.optimizer) * self.gamma)
        self.epoch += 1


def get_current_lr(optimizer):
    if hasattr(optimizer, 'param_groups'):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    else:
        return optimizer.lr


def set_current_lr(optimizer, lr):
    if hasattr(optimizer, 'param_groups'):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        optimizer.lr = lr

