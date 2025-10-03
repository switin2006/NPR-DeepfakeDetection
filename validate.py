import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from data import create_dataloader

def validate(model, opt):
    # Ensure the model is in evaluation mode
    model.eval() 
    
    # Create the data loader using the provided options
    data_loader = create_dataloader(opt)
    
    # --- Initialize loss function and accumulators ---
    loss_fn = nn.BCEWithLogitsLoss()
    total_val_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad(): # Disable gradient calculation for efficiency
        for i, data in enumerate(tqdm(data_loader, desc="Validating")):
            # --- Correctly unpack data as a tuple (image, label) ---
            img_batch, label_batch = data
            img_batch = img_batch.cuda()
            label_batch = label_batch.cuda().float() # Ensure label is float for loss

            # Pass the image batch directly through the model
            output_logits = model(img_batch)
            
            # --- Calculate and accumulate validation loss ---
            loss = loss_fn(output_logits.squeeze(1), label_batch)
            total_val_loss += loss.item() * img_batch.size(0)
            
            # --- Accumulate predictions and labels for other metrics ---
            probs = torch.sigmoid(output_logits).squeeze().cpu().numpy()
            labels = label_batch.cpu().numpy()

            if probs.ndim == 0:
                y_pred.append(probs)
            else:
                y_pred.extend(probs)
            
            if labels.ndim == 0:
                y_true.append(labels)
            else:
                y_true.extend(labels)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # --- Calculate the final average validation loss for the epoch ---
    avg_val_loss = total_val_loss / len(data_loader.dataset)

    # --- METRIC CALCULATIONS ---
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    preds = (y_pred > 0.5).astype(int)
    acc = np.mean(y_true == preds)
    
    epsilon = 1e-7
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    # --- RETURN THE 6 EXPECTED VALUES IN THE CORRECT ORDER ---
    return acc, ap, auc, precision, f1_score, avg_val_loss
