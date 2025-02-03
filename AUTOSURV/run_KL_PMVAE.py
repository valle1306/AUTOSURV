#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from KL_PMVAE import KL_PMVAE_Cancer
from utils import sort_data, load_data, bce_recon_loss, kl_divergence, get_match_id
from train_KL_PMVAE import train_KL_PMVAE
import os
# Ensure 'saved_models' directory exists
os.makedirs("saved_models", exist_ok=True)

dtype = torch.FloatTensor

# Parameters specific to your dataset
input_n1 = 810  # Update based on your radiomic features
input_n2 = 18   # Update based on your clinical features
z_dim = [8, 16, 32, 64]
EPOCH_NUM = [800, 1200, 1600]
NUM_CYCLES = [2, 4, 5]
CUTTING_RATIO = [0.3, 0.5, 0.7]
Initial_Learning_Rate = [0.05, 0.005, 0.001]
L2_Lambda = [0.1, 0.05, 0.005]

# Load training and validation data
_, x_train_radiomic = load_data("train_radiomic.csv", dtype)
_, x_valid_radiomic = load_data("test_radiomic.csv", dtype)


# Hyperparameter tuning
opt_l2 = 0
opt_lr = 0
opt_dim = 0
opt_epoch_num = 0
opt_num_cycle = 0
opt_cr = 0
opt_loss = torch.Tensor([float("Inf")])
if torch.cuda.is_available():
    opt_loss = opt_loss.cuda()

for l2 in L2_Lambda:
    for lr in Initial_Learning_Rate:
        for Z in z_dim:
            for Epoch_num in EPOCH_NUM:
                for Num_cycles in NUM_CYCLES:
                    for cutting_ratio in CUTTING_RATIO:
                        _, _, _, _, train_loss_unsup, eval_loss_unsup = train_KL_PMVAE(
                            x_train_radiomic, yevent_train,
                            x_valid_radiomic, yevent_valid,
                            Z, input_n1, input_n2,
                            lr, l2, cutting_ratio, Epoch_num, Num_cycles, dtype,
                            path="saved_models/unsup_checkpoint_tune.pt"
                        )
                        if eval_loss_unsup < opt_loss:
                            opt_l2 = l2
                            opt_lr = lr
                            opt_dim = Z
                            opt_epoch_num = Epoch_num
                            opt_num_cycle = Num_cycles
                            opt_cr = cutting_ratio
                            opt_loss = eval_loss_unsup
                        print(type(Z))
                        print(f"Epoch: {Epoch_num}, Cycles: {Num_cycles}, CR: {cutting_ratio:.4f}, "
      f"F1: {float(lr):.4f}, Z: {int(Z) if isinstance(Z, int) else Z.item()}, "
      f"Train loss: {train_loss_unsup.item():.4f}, Eval loss: {eval_loss_unsup.item():.4f}")



print(f"Optimal Epochs: {opt_epoch_num}, Cycles: {opt_num_cycle}, CR: {opt_cr:.4f}, "
      f"L2: {opt_l2:.4f}, LR: {opt_lr:.4f}, z_dim: {opt_dim}")


# Final training with optimal parameters
train_mean, train_logvar, test_mean, test_logvar, train_loss_unsup, test_loss_unsup = train_KL_PMVAE(
    x_train_radiomic, yevent_train,
    x_valid_radiomic, yevent_valid,
    opt_dim, input_n1, input_n2,
    opt_lr, opt_l2, opt_cr, opt_epoch_num, opt_num_cycle, dtype, save_model=True,
    path="saved_models/unsup_checkpoint_final.pt"
)


print(f"Final Training Loss: {train_loss_unsup[0].item():.4f}, Testing Loss: {test_loss_unsup.item():.4f}")

# Process and save latent representations
train_z = train_mean
test_z = test_mean

processed_train = torch.cat((ytime_train, yevent_train, train_z), 1)
processed_test = torch.cat((ytime_valid, yevent_valid, test_z), 1)

z_count = np.array(list(range(1, train_z.size()[1] + 1))).astype(str)
z_names = [f"Z_{i}" for i in z_count]

processed_train_df = pd.DataFrame(processed_train.detach().cpu().numpy(), columns=["PFS", "Censorship"] + z_names)
processed_test_df = pd.DataFrame(processed_test.detach().cpu().numpy(), columns=["PFS", "Censorship"] + z_names)

processed_train_df.to_csv("processed_train.csv", index=False)
processed_test_df.to_csv("processed_test.csv", index=False)


# %%



# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from KL_PMVAE import KL_PMVAE_Cancer
from utils import sort_data, load_data
from train_KL_PMVAE import train_KL_PMVAE
import os
import time

# Ensure 'saved_models' directory exists
os.makedirs("saved_models", exist_ok=True)

dtype = torch.FloatTensor

# Parameters specific to your radiomic dataset
input_n1 = 810  # Number of radiomic features
input_n2 = 0    # No clinical features for this step
z_dim = [8, 16, 32, 64]
EPOCH_NUM = [800, 1200, 1600]
NUM_CYCLES = [2, 4, 5]
CUTTING_RATIO = [0.3, 0.5, 0.7]
Initial_Learning_Rate = [0.05, 0.005, 0.001]
L2_Lambda = [0.1, 0.05, 0.005]

# Load training and validation radiomic data
_, x_train_radiomic = load_data("train_radiomic.csv", dtype)
_, x_valid_radiomic = load_data("test_radiomic.csv", dtype)

# Hyperparameter tuning
opt_l2 = 0
opt_lr = 0
opt_dim = 0
opt_epoch_num = 0
opt_num_cycle = 0
opt_cr = 0
opt_loss = torch.Tensor([float("Inf")])
if torch.cuda.is_available():
    opt_loss = opt_loss.cuda()

start_time = time.time()

# Perform hyperparameter tuning
for l2 in L2_Lambda:
    for lr in Initial_Learning_Rate:
        for Z in z_dim:
            for Epoch_num in EPOCH_NUM:
                for Num_cycles in NUM_CYCLES:
                    for cutting_ratio in CUTTING_RATIO:
                        mean, logvar, val_mean, val_logvar, train_loss_unsup, eval_loss_unsup = train_KL_PMVAE(
                            x_train_radiomic, x_valid_radiomic, Z, input_n1,
                            lr, l2, cutting_ratio, Epoch_num, Num_cycles, dtype,
                            save_model=False, path="saved_models/unsup_checkpoint_tune.pt")
                        
                        if eval_loss_unsup < opt_loss:
                            opt_l2 = l2
                            opt_lr = lr
                            opt_dim = Z
                            opt_epoch_num = Epoch_num
                            opt_num_cycle = Num_cycles
                            opt_cr = cutting_ratio
                            opt_loss = eval_loss_unsup

                        print(f"Epoch: {Epoch_num}, Cycles: {Num_cycles}, CR: {cutting_ratio:.4f}, "
                              f"L2: {l2:.4f}, LR: {lr:.4f}, z_dim: {Z}, "
                              f"Train loss: {train_loss_unsup.item():.4f}, Eval loss: {eval_loss_unsup.item():.4f}")

print(f"Optimal Epochs: {opt_epoch_num}, Cycles: {opt_num_cycle}, CR: {opt_cr:.4f}, "
      f"L2: {opt_l2:.4f}, LR: {opt_lr:.4f}, z_dim: {opt_dim}")
print(f"Tuning time: {time.time() - start_time:.2f} seconds")


# Final training with optimal parameters
train_mean, train_logvar, test_mean, test_logvar, train_loss_unsup, test_loss_unsup = train_KL_PMVAE(
    x_train_radiomic,  # Training radiomic data
    x_valid_radiomic,  # Validation radiomic data
    opt_dim,           # Optimal latent space dimension
    input_n1,          # Number of radiomic features
    opt_lr,            # Optimal learning rate
    opt_l2,            # Optimal L2 regular_ization
    opt_cr,            # Optimal cutting ratio
    opt_epoch_num,     # Optimal number of epochs
    opt_num_cycle,     # Optimal number of cycles
    dtype,             # Data type
    save_model=True,   # Save the model
    path="saved_models/unsup_checkpoint_final.pt"  # Save path for the final model
)

print(f"Final Training Loss: {train_loss_unsup.item():.4f}, Testing Loss: {test_loss_unsup.item():.4f}")

# Process and save latent representations
train_z = train_mean
test_z = test_mean

# Only save latent features from radiomic data
z_count = np.array(list(range(1, train_z.size()[1] + 1))).astype(str)
z_names = [f"Z_{i}" for i in z_count]

processed_train_df = pd.DataFrame(train_z.detach().cpu().numpy(), columns=z_names)
processed_test_df = pd.DataFrame(test_z.detach().cpu().numpy(), columns=z_names)

processed_train_df.to_csv("processed_train_radiomic.csv", index=True)
processed_test_df.to_csv("processed_test_radiomic.csv", index=True)


# %%

# %%
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from KL_PMVAE import KL_PMVAE_Cancer
from utils import sort_data, load_data
from train_KL_PMVAE import train_KL_PMVAE
import os
import time

# Ensure 'saved_models' directory exists
os.makedirs("saved_models", exist_ok=True)

dtype = torch.FloatTensor

# Parameters specific to your radiomic dataset
input_n1 = 810  # Number of radiomic features
input_n2 = 0    # No clinical features for this step
z_dim = [8, 16, 32, 64]
EPOCH_NUM = [800, 1200, 1600]
NUM_CYCLES = [2, 4, 5]
CUTTING_RATIO = [0.3, 0.5, 0.7]
Initial_Learning_Rate = [0.05, 0.005, 0.001]
L2_Lambda = [0.1, 0.05, 0.005]

# Load training and validation radiomic data
_, x_train_radiomic = load_data("train_radiomic.csv", dtype)
_, x_valid_radiomic = load_data("test_radiomic.csv", dtype)

# Hyperparameter tuning
opt_l2 = 0
opt_lr = 0
opt_dim = 0
opt_epoch_num = 0
opt_num_cycle = 0
opt_cr = 0
opt_loss = torch.Tensor([float("Inf")])
if torch.cuda.is_available():
    opt_loss = opt_loss.cuda()

start_time = time.time()

# Perform hyperparameter tuning
for l2 in L2_Lambda:
    for lr in Initial_Learning_Rate:
        for Z in z_dim:
            for Epoch_num in EPOCH_NUM:
                for Num_cycles in NUM_CYCLES:
                    for cutting_ratio in CUTTING_RATIO:
                        mean, logvar, val_mean, val_logvar, train_loss_unsup, eval_loss_unsup = train_KL_PMVAE(
                            x_train_radiomic, x_valid_radiomic, Z, input_n1,
                            lr, l2, cutting_ratio, Epoch_num, Num_cycles, dtype,
                            save_model=False, path="saved_models/unsup_checkpoint_tune.pt")
                        
                        if eval_loss_unsup < opt_loss:
                            opt_l2 = l2
                            opt_lr = lr
                            opt_dim = Z
                            opt_epoch_num = Epoch_num
                            opt_num_cycle = Num_cycles
                            opt_cr = cutting_ratio
                            opt_loss = eval_loss_unsup

                        print(f"Epoch: {Epoch_num}, Cycles: {Num_cycles}, CR: {cutting_ratio:.4f}, "
                              f"L2: {l2:.4f}, LR: {lr:.4f}, z_dim: {Z}, "
                              f"Train loss: {train_loss_unsup.item():.4f}, Eval loss: {eval_loss_unsup.item():.4f}")

print(f"Optimal Epochs: {opt_epoch_num}, Cycles: {opt_num_cycle}, CR: {opt_cr:.4f}, "
      f"L2: {opt_l2:.4f}, LR: {opt_lr:.4f}, z_dim: {opt_dim}")
print(f"Tuning time: {time.time() - start_time:.2f} seconds")

# Save optimal hyperparameters to a JSON file
optimal_params = {
    "opt_l2": opt_l2,
    "opt_lr": opt_lr,
    "opt_dim": opt_dim,
    "opt_epoch_num": opt_epoch_num,
    "opt_num_cycle": opt_num_cycle,
    "opt_cr": opt_cr
}

with open("saved_models/optimal_params.json", "w") as f:
    json.dump(optimal_params, f, indent=4)

# Final training with optimal parameters
train_mean, train_logvar, test_mean, test_logvar, train_loss_unsup, test_loss_unsup = train_KL_PMVAE(
    x_train_radiomic,  # Training radiomic data
    x_valid_radiomic,  # Validation radiomic data
    opt_dim,           # Optimal latent space dimension
    input_n1,          # Number of radiomic features
    opt_lr,            # Optimal learning rate
    opt_l2,            # Optimal L2 regularization
    opt_cr,            # Optimal cutting ratio
    opt_epoch_num,     # Optimal number of epochs
    opt_num_cycle,     # Optimal number of cycles
    dtype,             # Data type
    save_model=True,   # Save the model
    path="saved_models/unsup_checkpoint_final_1.pt"  # Save path for the final model
)

print(f"Final Training Loss: {train_loss_unsup.item():.4f}, Testing Loss: {test_loss_unsup.item():.4f}")

# Process and save latent representations
train_z = train_mean
test_z = test_mean

# Only save latent features from radiomic data
z_count = np.array(list(range(1, train_z.size()[1] + 1))).astype(str)
z_names = [f"Z_{i}" for i in z_count]

processed_train_df = pd.DataFrame(train_z.detach().cpu().numpy(), columns=z_names)
processed_test_df = pd.DataFrame(test_z.detach().cpu().numpy(), columns=z_names)

processed_train_df.to_csv("processed_train_radiomic.csv", index=True)
processed_test_df.to_csv("processed_test_radiomic.csv", index=True)


# %%
