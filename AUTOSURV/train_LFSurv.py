# +
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from lifelines.utils import concordance_index
from LFSurv import LFSurv
from utils import neg_par_log_likelihood, c_index, EarlyStopping
import pandas as pd
import os

dtype = torch.FloatTensor


def train_LFSurv(train_data, train_ytime, train_yevent,
                 eval_data, eval_ytime, eval_yevent,
                 latent_dim, clinical_dim, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, 
                 Learning_Rate, L2, epoch_num, patience, 
                 path="saved_model/sup_checkpoint.pt"):
    """
    Train the LFSurv model using separate latent and clinical features.

    Returns:
        train_y_pred: Predictions on training data.
        eval_y_pred: Predictions on evaluation data.
        train_cindex: C-index on training data.
        eval_cindex: C-index on evaluation data.
        best_epoch_num: Epoch number of the best model.
    """
    # Ensure save directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Define the total input size
    input_n = latent_dim + clinical_dim

    # Initialize the LFSurv model
    net = LFSurv(latent_dim, input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2)

    early_stopping_sup = EarlyStopping(patience=patience, verbose=False, path=path)

    if torch.cuda.is_available():
        net.cuda()
        train_data, train_ytime, train_yevent = train_data.cuda(), train_ytime.cuda(), train_yevent.cuda()
        eval_data, eval_ytime, eval_yevent = eval_data.cuda(), eval_ytime.cuda(), eval_yevent.cuda()

    # Split train_data and eval_data into latent and clinical features
    train_latent_features = train_data[:, :latent_dim]
    train_clinical_features = train_data[:, latent_dim:]
    eval_latent_features = eval_data[:, :latent_dim]
    eval_clinical_features = eval_data[:, latent_dim:]

    # Debugging: Print tensor shapes
    #print(f"Train latent features shape: {train_latent_features.shape}")
    #print(f"Train clinical features shape: {train_clinical_features.shape}")
    #print(f"Eval latent features shape: {eval_latent_features.shape}")
    #print(f"Eval clinical features shape: {eval_clinical_features.shape}")

    # Optimizer
    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)
    start_time = time.time()

    for epoch in range(epoch_num):
        # Training phase
        net.train()
        opt.zero_grad()
        y_pred = net(train_latent_features, train_clinical_features, s_dropout=True)  # Use separate features
        loss_sup = neg_par_log_likelihood(y_pred, train_ytime, train_yevent)
        loss_sup.backward()
        opt.step()
        # During training phase
        #print(f"Epoch {epoch + 1}: Loss={loss_sup.item()}")

        # Check predictions distribution
        #print(f"y_pred stats: min={y_pred.min().item()}, max={y_pred.max().item()}, mean={y_pred.mean().item()}, std={y_pred.std().item()}")

        # Validation phase
        net.eval()
        with torch.no_grad():
            eval_y_pred = net(eval_latent_features, eval_clinical_features, s_dropout=False)
            eval_cindex = c_index(eval_ytime, eval_yevent, eval_y_pred)
        #print(f"Eval y_pred stats: min={eval_y_pred.min().item()}, max={eval_y_pred.max().item()}, mean={eval_y_pred.mean().item()}, std={eval_y_pred.std().item()}")

        # Early stopping
        early_stopping_sup(eval_cindex, net)
        if early_stopping_sup.early_stop:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {early_stopping_sup.best_epoch_num}")
            break

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            train_y_pred = net(train_latent_features, train_clinical_features, s_dropout=False)
            train_cindex = c_index(train_ytime, train_yevent, train_y_pred)
            print(f"Epoch {epoch + 1}: Training C-index: {train_cindex:.4f}, Validation C-index: {eval_cindex:.4f}")

    # Load the best model
    print(f"Loading the best model from epoch {early_stopping_sup.best_epoch_num}")
    net.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))

    # Final evaluation
    net.eval()
    train_y_pred = net(train_latent_features, train_clinical_features, s_dropout=False)
    eval_y_pred = net(eval_latent_features, eval_clinical_features, s_dropout=False)
    train_cindex = c_index(train_ytime, train_yevent, train_y_pred)
    eval_cindex = c_index(eval_ytime, eval_yevent, eval_y_pred)
    print(f"Final Training C-index: {train_cindex:.4f}, Final Validation C-index: {eval_cindex:.4f}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

    return train_y_pred, eval_y_pred, train_cindex, eval_cindex, early_stopping_sup.best_epoch_num

# -

