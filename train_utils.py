# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:25:42 2025

@author: gaura
"""


import torch
from torch.utils.data import DataLoader
import time 
import copy
import itertools
import pickle
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os


def plot_losses(train_loss_record, val_loss_record, save_path=None):
    """
    Plot training and validation loss curves for total, cls, reg, and obj losses.
    Optionally save the plot to a file.
    
    Args:
    train_loss_record (dict): Dictionary with keys 'total', 'cls', 'reg', 'obj' and lists of loss values.
    val_loss_record (dict): Dictionary with the same structure as train_loss_record for validation losses.
    save_path (str, optional): Path to save the plot (e.g., 'loss_curves.png'). If None, plot is not saved.
    """
    # Get the number of epochs
    num_epochs = len(train_loss_record['total'])

    # Create epochs list
    epochs = list(range(1, num_epochs + 1))

    # Plot the training losses (solid lines)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_record['total'], label='Total Loss (Train)', linewidth=2)
    

    # Plot the validation losses (dashed lines)
    if len(val_loss_record['total']) > 0:
        plt.plot(epochs, val_loss_record['total'], label='Total Loss (Val)', linewidth=2, linestyle='--')
   

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Save the plot if save_path is provided
    if save_path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler , device, log_every=50, max_batches=None):
    """
    Train for one epoch, optionally limit number of batches (useful for quick tuning).
    Returns average training loss over processed batches.
    """
    model.train()
    total_loss = 0.0
    n = 0
    for i, (im, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        im = im.to(device)
        
        optimizer.zero_grad()
        
        #with autocast(device_type="cuda", dtype=torch.bfloat16):
        mu ,log_var , px = model(im)
        loss= loss_fn(mu, log_var ,px ,im)
        
        # Scaled backward + optimizer step
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        loss.backward()
        optimizer.step()

        total_loss += float(loss.cpu().item())
        n += 1

        if (i + 1) % log_every == 0:
            print(f"    batch {i+1} loss: {loss.item():.4f}")

    avg_loss = total_loss / max(1, n)
    
    return avg_loss


@torch.no_grad()
def evaluate_val_loss(model, val_loader, loss_fn, device, max_batches=None):
    """
    Compute average validation loss over at most max_batches batches.
    If max_batches is None -> run full val_loader (be careful, this may be slow).
    """
    model.eval()
    total_loss = 0.0
    
    n = 0
    for i, (im, _) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        im = im.to(device)
    
        #with autocast(device_type="cuda", dtype=torch.bfloat16):
        mu ,log_var , px = model(im)
        loss= loss_fn(mu, log_var ,px ,im)
        
        total_loss += float(loss.cpu().item())  # accumulate as float
        
        n += 1
        
    model.train()
    
    avg_loss = total_loss / max(1, n)
    
    return avg_loss

def train(model, train_loader, val_loader, loss_fn, optimizer, device,
          epochs=10, validate_every_epoch=True, val_batches_limit=None,
          train_batches_limit=None, best_val_loss = None ,checkpoint_path='best.pt', scheduler=None,
          early_stopping_patience=None, log_every=5, warmup = 1):
    """
    Train with optional validation each epoch.
    - train_batches_limit: int or None. If int, only this many batches per epoch (useful for quick runs).
    - val_batches_limit: int or None. If int, only this many val batches are used to compute val loss.
    - scheduler: learning rate scheduler (optional).
    - early_stopping_patience: if set, stops if val loss doesn't improve for this many epochs.
    """
    device = device
    model = model.to(device)
    
    if best_val_loss is None:
        best_val_loss = float('inf')
    else:
        best_val_loss = best_val_loss
        
    epochs_no_improve = 0
    best_model_state = None
    scaler = GradScaler()
    warmup_epochs = warmup
    target_lr = optimizer.param_groups[0]['lr']
    target_beta = loss_fn.beta
    train_loss_record = {'total':[]}
    val_loss_record = {'total':[]}
    val_loss = None

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}:")
        
        if epoch <= warmup_epochs:
            # Linear Ramp: 0 -> target_lr
            #current_lr = target_lr * (epoch / warmup_epochs)
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = current_lr
            
            # Linear Ramp: 0 -> target_beta
            current_beta = target_beta * ( (epoch-1) / warmup_epochs)
            loss_fn.beta = current_beta
            
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device,
                                     log_every=log_every, max_batches=train_batches_limit)
        t1 = time.time()
        print(f"  Train loss (avg over processed batches): {train_loss:.4f}  time: {t1 - t0:.1f}s")
        train_loss_record['total'].append(train_loss)
        

        if validate_every_epoch and val_loader is not None:
            val_loss = evaluate_val_loss(model, val_loader, loss_fn, device, max_batches=val_batches_limit)
            print(f"  Val loss (avg): {val_loss:.4f}")
            
            val_loss_record['total'].append(val_loss)
            
            # track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({'model_state': best_model_state,
                            'optimizer_state': optimizer.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss}, checkpoint_path)
                print(f"  --> New best model saved to {checkpoint_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping: no improvement in {epochs_no_improve} epochs.")
                break

        if scheduler is not None and epoch > warmup_epochs:
            # step scheduler (choose step per epoch or after validation as required)
            try:
                scheduler.step()
            except Exception:
                pass
        
        
        torch.save({'model_state':model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss}, r'checkpoints\current_model.pt')

    # restore best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model ,(train_loss_record, val_loss_record)




if __name__ == '__main__':
   pass
    
    
    
    


