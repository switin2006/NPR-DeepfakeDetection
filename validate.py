import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from data import create_dataloader

def validate(model, opt):
    # Ensure the model is in evaluation mode
    model.eval() 
    
    # Create the data loader using the provided options
    data_loader = create_dataloader(opt)
    
    y_true, y_pred = [], []
    
    with torch.no_grad(): # Disable gradient calculation for efficiency
        for i, data in enumerate(tqdm(data_loader, desc="Validating")):
            # ==========================================================================================
            # --- FIX: Unpack data as a tuple (image, label) instead of a dictionary ---
            # ==========================================================================================
            img_batch, label_batch = data
            img_batch = img_batch.cuda() # Move image to GPU

            # Pass the image batch directly through the model
            output_logits = model(img_batch)
            # ==========================================================================================
            
            # Convert predictions to probabilities using sigmoid
            probs = torch.sigmoid(output_logits).squeeze().cpu().detach().numpy()
            labels = label_batch.cpu().detach().numpy()

            # Handle cases where the batch size is 1, so probs is not an array
            if probs.ndim == 0:
                y_pred.append(probs)
            else:
                y_pred.extend(probs)
            
            if labels.ndim == 0:
                y_true.append(labels)
            else:
                y_true.extend(labels)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # --- METRIC CALCULATIONS (UNCHANGED) ---
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

    return acc, ap, auc, y_true, y_pred, precision, f1_score
