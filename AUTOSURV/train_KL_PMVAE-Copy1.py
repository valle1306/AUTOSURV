#!/usr/bin/env python
# coding: utf-8
# %%


from utils import bce_recon_loss, kl_divergence
from KL_PMVAE import KL_PMVAE_Cancer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from KL_PMVAE import KL_PMVAE_Cancer
from utils import sort_data, load_data, bce_recon_loss, kl_divergence, get_match_id

def train_KL_PMVAE(train_x1, train_x2, eval_x1, eval_x2,
                   z_dim, input_n1, input_n2,
                   Learning_Rate, L2, Cutting_Ratio, p1_epoch_num, num_cycles, dtype, save_model=False,
                   path="saved_model/unsup_checkpoint.pt"):
    # Initialize the KL_PMVAE_Cancer model
    input_n1 = train_x1.shape[-1]  # Number of features in x1
    input_n2 = train_x2.shape[-1]  # Number of features in x2
    level_2_dim = input_n1 // 2    # Dynamically compute level_2_dim

    net = KL_PMVAE_Cancer(z_dim=z_dim, input_n1=input_n1, input_n2=input_n2, level_2_dim=level_2_dim)
    train_x1[torch.isnan(train_x1)] = torch.nanmean(train_x1)  # Imputation

    train_real = torch.cat((train_x1, train_x2), 1)
    train_real = (train_real - train_real.min()) / (train_real.max() - train_real.min())

    #print(f"train_x1 shape: {train_x1.shape}, train_x2 shape: {train_x2.shape}")
    
    eval_real = torch.cat((eval_x1, eval_x2), 1)
    eval_real = (eval_real - eval_real.min()) / (eval_real.max() - eval_real.min())
    # Debugging input data statistics
    #print(f"train_x1 shape: {train_x1.shape}, train_x1 Min: {train_x1.min()}, Max: {train_x1.max()}, NaN: {torch.isnan(train_x1).any()}")
    #print(f"train_x2 shape: {train_x2.shape}, train_x2 Min: {train_x2.min()}, Max: {train_x2.max()}, NaN: {torch.isnan(train_x2).any()}")
    #print(f"eval_x1 shape: {eval_x1.shape}, eval_x1 Min: {eval_x1.min()}, Max: {eval_x1.max()}, NaN: {torch.isnan(eval_x1).any()}")
    #print(f"eval_x2 shape: {eval_x2.shape}, eval_x2 Min: {eval_x2.max()}, Max: {eval_x2.max()}, NaN: {torch.isnan(eval_x2).any()}")
    
    # New debug statement to find NaN locations
    #print(f"NaN locations in train_x1: {torch.nonzero(torch.isnan(train_x1), as_tuple=True)}")


    # Move the model to GPU if available
    if torch.cuda.is_available():
        net.cuda()
    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)
    cycle_iter = p1_epoch_num // num_cycles
    start_time = time.time()

    for epoch in range(p1_epoch_num):
        # Calculate the KL weight (beta) for cyclical annealing
        tmp = float(epoch % cycle_iter) / cycle_iter
        if tmp == 0:
            beta = 0.1
        elif tmp <= Cutting_Ratio:
            beta = tmp / Cutting_Ratio
        else:
            beta = 1
        
        # Training phase
        net.train()
        opt.zero_grad()
        
        mean, logvar, _, recon_x1 = net(train_x1, train_x2, s_dropout=True)
        recon_x = torch.cat((recon_x1, train_x2), 1)
        #print(f"Recon_x Min: {recon_x.min()}, Max: {recon_x.max()}, NaN: {torch.isnan(recon_x).any()}")
        #print(f"Train_real Min: {train_real.min()}, Max: {train_real.max()}, NaN: {torch.isnan(train_real).any()}")


        recon_loss = bce_recon_loss(recon_x, train_real)
        total_kld, _, _ = kl_divergence(mean, logvar)
        loss_unsup = recon_loss + beta * total_kld
        
        loss_unsup.backward()
        opt.step()
        
        # Validation and logging every 100 epochs
        if (epoch + 1) % 100 == 0:
            net.eval()
            
            # Training metrics
            train_mean, train_logvar, _, train_recon1 = net(train_x1, train_x2, s_dropout=False)
            train_recon = torch.cat((train_recon1, train_x2), 1)
            train_recon_loss = bce_recon_loss(train_recon, train_real)
            train_total_kld, _, _ = kl_divergence(train_mean, train_logvar)
            train_loss_unsup = train_recon_loss + beta * train_total_kld
            
            # Validation metrics
            eval_mean, eval_logvar, _, eval_recon1 = net(eval_x1, eval_x2, s_dropout=False)
            eval_recon = torch.cat((eval_recon1, eval_x2), 1)
            eval_recon_loss = bce_recon_loss(eval_recon, eval_real)
            eval_total_kld, _, _ = kl_divergence(eval_mean, eval_logvar)
            eval_loss_unsup = eval_recon_loss + beta * eval_total_kld
            
            temp_epoch = epoch + 1
            print(f"Epoch: {temp_epoch}, "
                  f"Loss in training: {np.array(train_loss_unsup.detach().cpu().numpy()).round(4)}, "
                  f"Loss in validation: {np.array(eval_loss_unsup.detach().cpu().numpy()).round(4)}.")
    
    # Save the model if required
    if save_model:
        print("Saving model...")
        torch.save(net.state_dict(), path)
        print("Model saved.")
    
    # Print training time
    print(f"Training completed in {np.array(time.time() - start_time).round(2)} seconds.")
    
    return train_mean, train_logvar, eval_mean, eval_logvar, train_loss_unsup, eval_loss_unsup


# %%
