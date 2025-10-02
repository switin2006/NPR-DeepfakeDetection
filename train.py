# This is the new, complete train.py adapted for CIFAKE with tqdm

import os
import sys
import time
import torch
import torch.nn
import numpy as np
from tensorboardX import SummaryWriter
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
import random
from tqdm import tqdm # <--- IMPORT TQDM

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_val_opt(opt):
    """Prepares options specifically for the validation set."""
    val_opt = TrainOptions().parse(print_options=False)
    # Important: Set the dataroot to your validation folder
    val_opt.dataroot = os.path.join(opt.dataroot, '..', opt.val_split) # Assumes 'val' is sibling to 'train'
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    
    # Set the dataroot directly to the training split
    opt.dataroot = os.path.join(opt.dataroot, opt.train_split)
    
    # Create the validation options based on the training options
    val_opt = get_val_opt(opt)
    print(f"--- Training data will be loaded from: {opt.dataroot}")
    print(f"--- Validation data will be loaded from: {val_opt.dataroot}")

    # Set up logging and data loader
    print(' '.join(list(sys.argv)))
    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    
    print(f'Starting training for {opt.niter} epochs...')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        
        # ==========================================================================================
        # WRAP THE DATA LOADER WITH TQDM
        # ==========================================================================================
        # --- TRAINING LOOP ---
        for i, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{opt.niter} [Train]")):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                # This print statement is fine, but tqdm will handle the main progress
                train_writer.add_scalar('loss', model.loss, model.total_steps)
        
        # --- Learning Rate Schedule ---
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(f'Adjusting learning rate at the end of epoch {epoch}')
            model.adjust_learning_rate()
            
        # ==========================================================================================
        # VALIDATION AFTER EVERY EPOCH
        # ==========================================================================================
        print(f'--- Running validation for epoch {epoch+1} ---')
        model.eval() # Set model to evaluation mode
        
        # The validate function from validate.py is called here
        acc, ap = validate(model.model, val_opt)[:2]
        
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        
        print(f"(Val @ epoch {epoch+1}) Accuracy: {acc:.4f}; AP: {ap:.4f}")
        
        model.train() # Set model back to training mode
        # ==========================================================================================

    # --- FINAL ACTIONS ---
    print('Training finished. Saving the last model.')
    model.save_networks('last')
    
