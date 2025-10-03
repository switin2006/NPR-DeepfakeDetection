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
from tqdm import tqdm

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

def get_val_opt(opt):
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = os.path.join(opt.dataroot, '..', opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt

if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    
    opt.dataroot = os.path.join(opt.dataroot, opt.train_split)
    val_opt = get_val_opt(opt)
    print(f"--- Training data will be loaded from: {opt.dataroot}")
    print(f"--- Validation data will be loaded from: {val_opt.dataroot}")

    print(' '.join(list(sys.argv)))
    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    
    # --- INITIALIZE FOR EARLY STOPPING BASED ON LOSS ---
    best_val_loss = float('inf') # Initialize with a high value
    patience_counter = 0
    # ---------------------------------------------------

    print(f'Starting training for {opt.niter} epochs...')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        
        # --- TRAINING LOOP ---
        model.train()
        total_train_loss = 0.0 # Accumulator for training loss
        
        for i, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{opt.niter} [Train]")):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()
            
            batch_loss = model.loss.item()
            total_train_loss += batch_loss * data[0].size(0) # Multiply by batch size
            
            if model.total_steps % opt.loss_freq == 0:
                train_writer.add_scalar('loss/batch_loss', batch_loss, model.total_steps)
        
        avg_train_loss = total_train_loss / len(data_loader.dataset)
        train_writer.add_scalar('loss/avg_epoch_loss', avg_train_loss, epoch)
        
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(f'Adjusting learning rate at the end of epoch {epoch}')
            model.adjust_learning_rate()
            
        # --- VALIDATION AFTER EVERY EPOCH ---
        print(f'--- Running validation for epoch {epoch+1} ---')
        model.eval()
        
        # --- UNPACK NEW RETURN VALUES, INCLUDING val_loss ---
        acc, ap, auc, precision, f1_score, val_loss = validate(model.model, val_opt)
        
        val_writer.add_scalar('loss/validation_loss', val_loss, epoch)
        val_writer.add_scalar('metrics/accuracy', acc, epoch)
        val_writer.add_scalar('metrics/ap', ap, epoch)
        
        # --- PRINT UPDATED LOGS WITH LOSS ---
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {acc:.4f} | Val AP: {ap:.4f} | Val F1: {f1_score:.4f}")
        
        # --- EARLY STOPPING LOGIC BASED ON VALIDATION LOSS ---
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model.")
            best_val_loss = val_loss
            model.save_networks('best')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{opt.earlystop_epoch}")

        if patience_counter >= opt.earlystop_epoch:
            print(f"Early stopping triggered after {opt.earlystop_epoch} epochs without improvement.")
            break
            
    print('Training finished. Saving the last model.')
    model.save_networks('last')
    
