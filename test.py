import os
import torch
import numpy as np
from validate import validate
from networks.trainer import Trainer
from options.test_options import TestOptions
import random

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    seed_torch(100)
    opt = TestOptions().parse()

    opt.isTrain = False 
    model = Trainer(opt)
    model.load_networks(opt.resume)
    
    print(f"--- Evaluating model: {opt.resume} ---")
    print(f"--- On test dataset: {opt.dataroot} ---")

    model.eval()
    
    # Unpack the new return values from the validate function
    acc, ap, auc, _, _, precision, f1_score = validate(model.model, opt)

    print('*' * 35)
    print(f"Evaluation Results for CIFAKE Test Set:")
    print(f"  Accuracy:           {acc*100:.2f}%")
    print(f"  Average Precision:  {ap*100:.2f}%")
    print(f"  Precision:          {precision*100:.2f}%")
    print(f"  F1-Score:           {f1_score*100:.2f}%")
    print('*' * 35)

