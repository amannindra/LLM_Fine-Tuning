
import torch
import json
from pathlib import Path

import numpy as np
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"

R = "The cat is on the mat."
C = "The cat and the dog."

def rouge_1_precision(R, C):
    dicts = {}
    count = 0
    for i in R.split():
        if i in dicts:
            dicts[i] += 1
        else:
            dicts[i] = 1
    for i in C.split():
        if i in dicts:
            count += 1
    return count 

print(f"Total count: {rouge_1_precision(R, C)}")
        
