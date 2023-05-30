import numpy as np
import torch
import re
import os

def get_optimizer(name, params):
    if name == "Adam":
        return lambda model_params: torch.optim.Adam(model_params, lr=params["lr"], weight_decay=params["weight_decay"])
    else:
        raise ValueError("Unknown optimizer name: {}".format(name))


def parse_float_arg(input, prefix):
    p = re.compile(prefix+"_[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    if m is None:
        return None
    input = m.group()
    p = re.compile("[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    return float(m.group())


class ScalarMovingAverage:

    def __init__(self, eps=0):
        self.len_sum = 0
        self.len_count = eps

    def add(self, value, slen):
        self.len_sum += slen*value
        self.len_count += slen
        return self
    
    def get_len_weight(self):
        return self.len_sum / self.len_count