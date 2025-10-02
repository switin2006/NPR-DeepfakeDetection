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
    torch.cuda.manual_seed_all(torch.cuda.device_count() > 1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    seed_torch(100)
    opt = TestOptions().parse()

    # We still need to set this for the Trainer's internal setup
    opt.isTrain = False 
    original_epoch = opt.epoch
    opt.epoch = 'dummy_epoch_that_will_not_be_found'
    model = Trainer(opt) 
    opt.epoch = original_epoch # Restore original epoch if needed elsewhere
    print(f"--- Manually loading model state from: {opt.resume} ---")
    if not os.path.exists(opt.resume):
        raise FileNotFoundError(f"FATAL: The specified model path does not exist: {opt.resume}")
        
    # The saved file is a dictionary containing the model's state.
    state_dict = torch.load(opt.resume, map_location=model.device)
    
    # 3. Apply the loaded weights directly to the model inside the Trainer.
    #    The actual network is stored in `model.model`.
    model.model.load_state_dict(state_dict)
    
    # ==========================================================================================
    
    print(f"--- Evaluating model: {opt.resume} ---")
    print(f"--- On test dataset: {opt.dataroot} ---")

    model.eval() # Ensure model is in evaluation mode
    
    # Call validate function
    acc, ap, auc, _, _, precision, f1_score = validate(model.model, opt)

    print('*' * 35)
    print(f"Evaluation Results for CIFAKE Test Set:")
    print(f"  Accuracy:           {acc*100:.2f}%")
    print(f"  Average Precision:  {ap*100:.2f}%")
    print(f"  Precision:          {precision*100:.2f}%")
    print(f"  F1-Score:           {f1_score*100:.2f}%")
    print('*' * 35)
