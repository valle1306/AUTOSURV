#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# !/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv


dtype = torch.FloatTensor

"""
Dataloaders
"""

def sort_data(path):
    """
    Sort the data based on survival time and extract features and labels.
    """
    data = pd.read_csv(path)
    
    # Check for columns in the dataset to distinguish between clinical and radiomic
    if "Progression Free Survival" in data.columns and "Censorship" in data.columns:
        # Clinical dataset
        print("Processing clinical dataset...")

        # Sort by "Progression Free Survival" (if required)
        data.sort_values("Progression Free Survival", ascending=False, inplace=True)

        # Extract subject ID
        subject_id = data.loc[:, ["SubjectID"]]

        # Extract features (excluding ID, PFS, and Censorship)
        x = data.drop(["SubjectID", "Progression Free Survival", "Censorship"], axis=1).values
        
        # Extract labels
        ytime = data.loc[:, ["Progression Free Survival"]].values  # Survival time
        yevent = data.loc[:, ["Censorship"]].values  # Event indicators

        return subject_id, x, ytime, yevent

    else:
        # Radiomic dataset
        print("Processing radiomic dataset...")

        # Extract subject ID
        subject_id = data.loc[:, ["SubjectID"]]

        # Use all other columns as features
        x = data.drop(["SubjectID"], axis=1).values
        
        # Return only features and ID for radiomic data
        return subject_id, x

def load_data(path, dtype):
    """
    Load data and convert it to PyTorch tensors.
    """
    # Extract clinical or radiomic data
    data = sort_data(path)

    if len(data) == 4:  # Clinical dataset (with ytime and yevent)
        subject_id, x, ytime, yevent = data

        # Convert to PyTorch tensors
        X = torch.from_numpy(x).type(dtype)
        YTIME = torch.from_numpy(ytime).type(dtype)
        YEVENT = torch.from_numpy(yevent).type(dtype)

        # Move to GPU if available
        if torch.cuda.is_available():
            X = X.cuda()
            YTIME = YTIME.cuda()
            YEVENT = YEVENT.cuda()

        return subject_id, X, YTIME, YEVENT

    elif len(data) == 2:  # Radiomic dataset (without ytime and yevent)
        subject_id, x = data

        # Convert to PyTorch tensors
        X = torch.from_numpy(x).type(dtype)

        # Move to GPU if available
        if torch.cuda.is_available():
            X = X.cuda()

        return subject_id, X


"""
Loss function for KL-PMVAE
"""

def bce_recon_loss(recon_x, x):
    """
    Binary Cross-Entropy reconstruction loss.
    """
    batch_size = x.size(0)
    assert batch_size != 0
    bce_loss = F.binary_cross_entropy(recon_x, x, reduction='sum').div(batch_size)
    return bce_loss

def kl_divergence(mu, logvar):
    """
    KL Divergence for latent variables.
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

"""
Loss function for LFSurv & C-index calculation
"""

def R_set(x):
    """
    Create the risk set indicator matrix.
    """
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix

def neg_par_log_likelihood(pred, ytime, yevent):
    """
    Negative partial log-likelihood for survival analysis.
    """

    # Ensure tensors have correct shapes
    #print(f"Shape of pred before reshape: {pred.shape}")
    if pred.dim() > 2 or pred.shape[1] != 1:
        raise ValueError("Expected pred to have shape (batch_size, 1)")
        
    pred = pred.view(-1, 1)  # Shape: (batch_size, 1)
    yevent = yevent.view(-1, 1)  # Shape: (batch_size, 1)

    # Generate risk set indicator
    ytime_indicator = R_set(ytime)  # Shape: (batch_size, batch_size)
    #print(f"ytime_indicator shape: {ytime_indicator.shape}")
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()

    # Compute risk set sum
    risk_set_sum = ytime_indicator.mm(torch.exp(pred))  # Shape: (batch_size, 1)
    # Compute partial log-likelihood
    diff = pred - torch.log(risk_set_sum)  # Shape: (batch_size, 1)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)  # Shape: (1, 1)
    n_observed = yevent.sum(0)
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))  # Shape: (1,)
    return cost


from lifelines.utils import concordance_index

def c_index(true_T, true_E, pred_risk):
    """
    Calculate concordance index for survival prediction using lifelines.
    """
    return concordance_index(true_T.detach().cpu().numpy(),
                             -pred_risk.detach().cpu().numpy(),
                             event_observed=true_E.detach().cpu().numpy())


"""
Early stopping scheme when training LFSurv
"""

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, path="saved_model/sup_checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.epoch_count = 0
        self.best_epoch_num = 1
        self.early_stop = False
        self.max_acc = None
        self.delta = delta
        self.path = path

    def __call__(self, acc, model):
        if self.max_acc is None:
            self.epoch_count += 1
            self.best_epoch_num = self.epoch_count
            self.max_acc = acc
            self.save_checkpoint(model)
        elif acc < self.max_acc + self.delta:
            self.epoch_count += 1
            self.counter += 1
            if self.counter % 20 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.epoch_count += 1
            self.best_epoch_num = self.epoch_count
            self.max_acc = acc
            if self.verbose:
                print(f'Validation accuracy increased ({self.max_acc:.6f} --> {acc:.6f}).  Saving model ...')
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

"""
Match patient IDs
"""

def get_match_id(id_1, id_2):
    """
    Match patient IDs between two datasets.
    """
    match_id_cache = []
    for i in id_2["SubjectID"]:
        match_id = id_1.index[id_1["SubjectID"] == i].tolist()
        match_id_cache += match_id
    return match_id_cache

"""
Sample from the groups
"""

def splitExprandSample(condition, sampleSize, expr):
    """
    Sample a subset from a group based on a condition.
    """
    split_expr = expr[condition].T
    split_expr = split_expr.sample(n=sampleSize, axis=1).T
    return split_expr


def calculate_harrell_cindex(y_time, y_event, y_pred):
    """
    Calculate Harrell's C-index.
    """
    return concordance_index(y_time, -y_pred, y_event)

def calculate_unos_cindex(y_time, y_event, y_pred):
    """
    Calculate Uno's C-index.
    """
    surv_data = Surv.from_arrays(event=y_event, time=y_time)
    return concordance_index_ipcw(surv_data, surv_data, -y_pred)[0]

def calculate_integrated_brier_score(y_time, y_event, y_pred, time_grid=None):
    """
    Calculate Integrated Brier Score (IBS).
    """
    surv_data = Surv.from_arrays(event=y_event, time=y_time)
    if time_grid is None:
        time_grid = np.linspace(y_time.min(), y_time.max(), 100)
    surv_probs = np.exp(-np.exp(-y_pred))  # Example of converting log-hazard to survival probabilities
    return integrated_brier_score(surv_data, surv_data, surv_probs, time_grid)



# %%



