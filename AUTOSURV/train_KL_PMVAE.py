#!/usr/bin/env python
# coding: utf-8
# %%

from utils import bce_recon_loss, kl_divergence
from KL_PMVAE import KL_PMVAE_Cancer
import torch
import torch.nn.functional as F
import numpy as np
import time

def train_KL_PMVAE(train_x1, eval_x1, z_dim, input_n1, Learning_Rate, L2, 
                   Cutting_Ratio, pl_epoch_num, num_cycles, dtype, save_model=False, path="saved_models/unsup_checkpoint.pt"):
    """
    Train the KL-PMVAE model specifically for radiomic data.
    """
    # Initialize the KL_PMVAE model
    input_n1 = train_x1.shape[1]  # Number of features in radiomic data
    level_2_dim = input_n1 // 2  # Dynamically compute level_2_dim

    net = KL_PMVAE_Cancer(z_dim=z_dim, input_n1=input_n1, input_n2=0, level_2_dim=level_2_dim)  # No clinical data
    train_x1 = torch.nan_to_num(train_x1)  # Handle NaN values in train_x1
    eval_x1 = torch.nan_to_num(eval_x1)  # Handle NaN values in eval_x1

    # Normalize data
    train_real = (train_x1 - train_x1.min()) / (train_x1.max() - train_x1.min())
    eval_real = (eval_x1 - eval_x1.min()) / (eval_x1.max() - eval_x1.min())

    # Move model to GPU if available
    if torch.cuda.is_available():
        net.cuda()
        train_x1, eval_x1 = train_x1.cuda(), eval_x1.cuda()
        train_real, eval_real = train_real.cuda(), eval_real.cuda()

    opt = torch.optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)
    cycle_iter = pl_epoch_num // num_cycles
    start_time = time.time()

    for epoch in range(pl_epoch_num):
        # Calculate the KL weight (beta) for cyclical annealing
        tmp = float(epoch % cycle_iter) / cycle_iter
        beta = tmp * Cutting_Ratio if tmp <= 0.8 else Cutting_Ratio

        # Training phase
        net.train()
        opt.zero_grad()

        mean, logvar, z, recon_x = net(train_x1, s_dropout=True)
        recon_loss = bce_recon_loss(recon_x, train_real)
        total_kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss_unsup = recon_loss + beta * total_kld

        loss_unsup.backward()
        opt.step()

        # Validation phase
        if (epoch + 1) % 100 == 0:
            net.eval()
            with torch.no_grad():
                val_mean, val_logvar, val_z, val_recon = net(eval_x1, s_dropout=False)
                val_recon_loss = bce_recon_loss(val_recon, eval_real)
                val_total_kld = -0.5 * torch.sum(1 + val_logvar - val_mean.pow(2) - val_logvar.exp())
                val_loss_unsup = val_recon_loss + beta * val_total_kld

            print(f"Epoch: {epoch + 1}, Train Loss: {loss_unsup.item():.4f}, "
                  f"Val Loss: {val_loss_unsup.item():.4f}")

    # Save model if required
    if save_model:
        print("Saving model...")
        torch.save(net.state_dict(), path)
        print("Model saved.")

    print(f"Training completed in {np.array(time.time() - start_time).round(2)} seconds.")

    return mean, logvar, val_mean, val_logvar, loss_unsup, val_loss_unsup


# %%
