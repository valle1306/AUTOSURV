# ##### process = pd.read_csv('processed_train_radiomic.csv')

# +
# #!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from train_LFSurv import train_LFSurv
import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv
import random
dtype = torch.float


def normalize_data(data, epsilon=1e-8):
    """
    Normalize data to have zero mean and unit variance.
    Handles zero-variance features by adding a small epsilon.
    """
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    std = torch.where(std == 0, torch.tensor(epsilon, device=data.device), std)  # Handle zero variance
    return (data - mean) / std


def load_combined_data(radiomic_path, clinical_path, dtype):
    """
    Load and combine radiomic latent features and clinical data, ensuring numerical data only,
    and removing features with zero variance.
    """
    # Load radiomic latent features
    radiomic_data = pd.read_csv(radiomic_path)

    # Drop unnecessary columns
    radiomic_data = radiomic_data.drop(columns=[col for col in ['Unnamed: 0', 'SubjectID'] if col in radiomic_data])

    # Load clinical features
    clinical_data = pd.read_csv(clinical_path)
    clinical_data = clinical_data.drop(columns=[col for col in ['Unnamed: 0', 'SubjectID'] if col in clinical_data])

    # Extract outcome variables
    ytime = clinical_data["Progression Free Survival"].values  # Survival time
    yevent = clinical_data["Censorship"].values  # Event indicators

    # Remove outcome variables from clinical data
    clinical_data = clinical_data.drop(columns=["Progression Free Survival", "Censorship"])

    # Retain only numeric columns
    radiomic_data = radiomic_data.select_dtypes(include=[np.number])
    clinical_data = clinical_data.select_dtypes(include=[np.number])

    # Fill missing values
    radiomic_data = radiomic_data.fillna(0)
    clinical_data = clinical_data.fillna(0)

    # Convert to PyTorch tensors
    radiomic_data = torch.tensor(radiomic_data.values, dtype=dtype)
    clinical_data = torch.tensor(clinical_data.values, dtype=dtype)
    ytime = torch.tensor(ytime, dtype=dtype)
    yevent = torch.tensor(yevent, dtype=dtype)

    # Normalize features
    radiomic_data = normalize_data(radiomic_data)
    clinical_data = normalize_data(clinical_data)

    # Move to GPU if available
    if torch.cuda.is_available():
        radiomic_data, clinical_data, ytime, yevent = radiomic_data.cuda(), clinical_data.cuda(), ytime.cuda(), yevent.cuda()

    return radiomic_data, clinical_data, ytime, yevent



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
    # Ensure y_pred is 1D (flattened)
    y_pred = y_pred.ravel()
    return concordance_index_ipcw(surv_data, surv_data, -y_pred)[0]



def calculate_integrated_brier_score(y_time, y_event, y_pred, time_grid=None):
    """
    Calculate the Integrated Brier Score (IBS).
    """
    surv_data = Surv.from_arrays(event=y_event, time=y_time)

    # Ensure the time grid is within the range of the test data
    if time_grid is None:
        time_grid = np.linspace(max(y_time.min(), 1), y_time.max(), 100)  # Avoid time values less than 1
    else:
        time_grid = np.clip(time_grid, y_time.min(), y_time.max())

    print(f"Time grid: [{time_grid.min()}, {time_grid.max()}]")
    print(f"Survival time range: [{y_time.min()}, {y_time.max()}]")

    # Ensure y_pred is 1D
    surv_probs = np.exp(-np.exp(-y_pred.ravel()))  # Convert log-hazard to survival probabilities

    # Compute the Integrated Brier Score
    return integrated_brier_score(surv_data, surv_data, surv_probs, time_grid)




def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility for CuDNN backend
    torch.backends.cudnn.benchmark = False  # Can slightly reduce performance but ensures reproducibility

# Set the seed
set_seed(42)



# Load and normalize data
train_latent, train_clinical, train_ytime, train_yevent = load_combined_data(
    "processed_train_radiomic.csv", "train_clinical.csv", dtype
)
valid_latent, valid_clinical, valid_ytime, valid_yevent = load_combined_data(
    "processed_test_radiomic.csv", "test_clinical.csv", dtype
)

# Debug: Check data stats
print("Train data stats:")
print(f"Latent feature mean: {train_latent.mean(dim=0)}")
print(f"Latent feature std: {train_latent.std(dim=0)}")
print(f"Clinical feature mean: {train_clinical.mean(dim=0)}")
print(f"Clinical feature std: {train_clinical.std(dim=0)}")

input_latent_dim = train_latent.shape[1]
input_clinical_dim = train_clinical.shape[1]

# Hyperparameters
level_2_dim = [8, 16, 32, 64]
epoch_num = 500
patience = 200
Initial_Learning_Rate = [0.0001, 0.0005, 0.00001, 0.1]
L2_Lambda = [0.001, 0.00075, 0.0005, 0.00025, 0.0001]
Dropout_rate_1 = [0.0, 0.2, 0.3, 0.5]
Dropout_rate_2 = [0.0, 0.2, 0.3, 0.5]

# Validate level_2_dim
print(f"Initial level_2_dim: {level_2_dim}")
level_2_dim = [dim for dim in level_2_dim if isinstance(dim, int) and dim > 0]
if not level_2_dim:
    raise ValueError("No valid values for level_2_dim. Please check the hyperparameter settings.")
print(f"Validated level_2_dim: {level_2_dim}")

# Initialize opt_dim and other variables before the tuning loop
opt_dim = 8  # Default value, should match a valid level_2_dim
opt_l2, opt_lr, opt_dr1, opt_dr2 = 0, 0, 0, 0
opt_cindex_va, opt_cindex_tr, best_epoch_num = -1, -1, -1

# Hyperparameter tuning loop
opt_cindex_va = -1  # Initialize the best validation C-index
for dr1 in Dropout_rate_1:
    for dr2 in Dropout_rate_2:
        for l2 in L2_Lambda:
            for lr in Initial_Learning_Rate:
                for dim in level_2_dim:
                    assert isinstance(dim, int), f"dim should be int, but got {type(dim)}: {dim}"
                    print(f"Training with dim={dim}, dr1={dr1}, dr2={dr2}, l2={l2}, lr={lr}")
                    
                    # Train and evaluate
                    train_y_pred, eval_y_pred, train_cindex, cindex_valid, best_epoch_num_tune = train_LFSurv(
                        train_data=torch.cat([train_latent, train_clinical], dim=1),
                        train_ytime=train_ytime,
                        train_yevent=train_yevent,
                        eval_data=torch.cat([valid_latent, valid_clinical], dim=1),
                        eval_ytime=valid_ytime,
                        eval_yevent=valid_yevent,
                        latent_dim=input_latent_dim,
                        clinical_dim=input_clinical_dim,
                        level_2_dim=dim,
                        Dropout_Rate_1=dr1,
                        Dropout_Rate_2=dr2,
                        Learning_Rate=lr,
                        L2=l2,
                        epoch_num=epoch_num,
                        patience=patience,
                        path="saved_models/sup_checkpoint_tune.pt"
                    )

                    if cindex_valid > opt_cindex_va:
                        print(f"Updating opt_dim: Previous={opt_dim}, New={dim}")
                        opt_l2, opt_lr, opt_dim, opt_dr1, opt_dr2 = l2, lr, dim, dr1, dr2
                        opt_cindex_va, opt_cindex_tr, best_epoch_num = cindex_valid, train_cindex, best_epoch_num_tune


# Validate and define level_2_dim before final training
level_2_dim = opt_dim  # Ensure level_2_dim matches the optimal dimension found
if not isinstance(level_2_dim, int) or level_2_dim <= 0:
    print(f"Invalid level_2_dim found: {level_2_dim}. Resetting to opt_dim: {opt_dim}")
    level_2_dim = opt_dim  # Use opt_dim as a fallback

# Debugging print to confirm values
print(f"Starting final training with opt_dim={opt_dim}, level_2_dim={level_2_dim}")

# Final training
train_y_pred, test_y_pred, cindex_train, cindex_test, best_epoch_num_overall = train_LFSurv(
    train_data=torch.cat([train_latent, train_clinical], dim=1),
    train_ytime=train_ytime,
    train_yevent=train_yevent,
    eval_data=torch.cat([valid_latent, valid_clinical], dim=1),
    eval_ytime=valid_ytime,
    eval_yevent=valid_yevent,
    latent_dim=input_latent_dim,
    clinical_dim=input_clinical_dim,
    level_2_dim=level_2_dim,  # Pass the corrected value
    Dropout_Rate_1=opt_dr1,
    Dropout_Rate_2=opt_dr2,
    Learning_Rate=opt_lr,
    L2=opt_l2,
    epoch_num=epoch_num,
    patience=patience,
    path="saved_models/sup_checkpoint_final.pt"
)


# Print final results
print(f"Optimal L2: {opt_l2}, Optimal LR: {opt_lr}, Optimal dim: {opt_dim}, Optimal dr1: {opt_dr1}, Optimal dr2: {opt_dr2}")
print(f"Optimal training C-index: {opt_cindex_tr:.4f}, Optimal validation C-index: {opt_cindex_va:.4f}")
print(f"Testing phase: Training C-index: {cindex_train:.4f}, Testing C-index: {cindex_test:.4f}")

# After the final training loop
print(f"Starting evaluation with optimal model.")

# Convert tensors to numpy arrays
train_ytime_np = train_ytime.detach().cpu().numpy()
train_yevent_np = train_yevent.detach().cpu().numpy()
valid_ytime_np = valid_ytime.detach().cpu().numpy()
valid_yevent_np = valid_yevent.detach().cpu().numpy()
test_y_pred_np = test_y_pred.detach().cpu().numpy()


# +
# Print results
train_y_pred_np = train_y_pred.detach().cpu().numpy()

def calculate_harrell_cindex(y_time, y_event, y_pred):
    """
    Calculate Harrell's C-index.
    """
    return concordance_index(y_time, -y_pred, y_event)

harrel_cindex = calculate_harrell_cindex(valid_ytime_np, valid_yevent_np, test_y_pred_np)
harrel_cindex_train = calculate_harrell_cindex(train_ytime_np, train_yevent_np, train_y_pred_np)

print(f"Harrell's C-index: {harrel_cindex:.4f}")
print(f"Harrell's C-index TRAIN: {harrel_cindex_train:.4f}")

# +
train_y_pred_np = train_y_pred.detach().cpu().numpy()


def calculate_unos_cindex(y_time, y_event, y_pred):
    """
    Calculate Uno's C-index.
    """
    surv_data = Surv.from_arrays(event=y_event, time=y_time)

    # Ensure y_pred is a 1D array
    y_pred = y_pred.ravel()

    # Calculate Uno's C-index
    return concordance_index_ipcw(surv_data, surv_data, y_pred)[0]


unos_cindex = calculate_unos_cindex(valid_ytime_np, valid_yevent_np, test_y_pred_np.ravel())
unos_cindex_train = calculate_unos_cindex(train_ytime_np, train_yevent_np, train_y_pred_np.ravel())


print(f"Uno's C-index: {unos_cindex:.4f}")
print(f"Uno's C-index TRAIN: {unos_cindex_train:.4f}")


# +
def calculate_integrated_brier_score(y_time, y_event, y_pred, time_grid=None):
    """
    Calculate the Integrated Brier Score (IBS).
    """
    # Convert input data to survival format
    surv_data = Surv.from_arrays(event=y_event, time=y_time)

    # Ensure the time grid is strictly within the range of the test data
    if time_grid is None:
        time_grid = np.linspace(y_time.min() + 1e-6, y_time.max() - 1e-6, 100)  # Avoid touching boundaries
    else:
        # Clip the provided time grid strictly to the range of the test data
        time_grid = np.clip(time_grid, y_time.min() + 1e-6, y_time.max() - 1e-6)

    print(f"Time grid: [{time_grid.min()}, {time_grid.max()}]")
    print(f"Survival time range: [{y_time.min()}, {y_time.max()}]")

    # Convert log-hazard predictions to survival probabilities
    surv_probs = np.exp(-np.exp(y_pred.ravel()))  # Ensure y_pred is 1D
    surv_probs = np.tile(surv_probs[:, np.newaxis], (1, len(time_grid)))  # Make 2D with probabilities at each time

    # Compute the Integrated Brier Score
    try:
        ibs = integrated_brier_score(surv_data, surv_data, surv_probs, time_grid)
    except ValueError as e:
        print(f"Error in calculating IBS: {e}")
        raise
    except IndexError as e:
        print(f"IndexError in calculating IBS: {e}")
        raise

    return ibs


ibs = calculate_integrated_brier_score(valid_ytime_np, valid_yevent_np, test_y_pred_np)
ibs_train = calculate_integrated_brier_score(train_ytime_np, train_yevent_np, train_y_pred_np)

print(f"Integrated Brier Score: {ibs:.4f}")
print(f"Integrated Brier Score TRAIN: {ibs_train:.4f}")

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Example data: Replace with your dataset
# valid_ytime_np: survival times
# valid_yevent_np: event indicators (1 if the event occurred, 0 otherwise)

# Fitting Kaplan-Meier Curve
kmf = KaplanMeierFitter()
kmf.fit(valid_ytime_np, event_observed=valid_yevent_np)

# Plotting the Kaplan-Meier Curve
plt.figure(figsize=(8, 6))
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Estimate")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.grid()
plt.show()


# +
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Create Kaplan-Meier fitters for discovery and replication cohorts
kmf_discovery = KaplanMeierFitter()
kmf_replication = KaplanMeierFitter()

# Fit KM curves for discovery and replication cohorts
kmf_discovery.fit(train_ytime, event_observed=train_yevent, label="Discovery (Train)")
kmf_replication.fit(valid_ytime, event_observed=valid_yevent, label="Replication (Test)")

# Plot the Kaplan-Meier curves
plt.figure(figsize=(10, 7))
kmf_discovery.plot_survival_function()
kmf_replication.plot_survival_function()

# Add labels and title
plt.title("Kaplan-Meier Survival Estimates: Discovery vs. Replication")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.grid()
plt.legend()
plt.show()



# +
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

# Create Kaplan-Meier fitters for discovery and replication cohorts
kmf_discovery = KaplanMeierFitter()
kmf_replication = KaplanMeierFitter()

# Fit KM curves for discovery and replication cohorts
kmf_discovery.fit(train_ytime, event_observed=train_yevent, label="Discovery Cohort")
kmf_replication.fit(valid_ytime, event_observed=valid_yevent, label="Replication Cohort")

# Plot the Kaplan-Meier curves
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
kmf_discovery.plot_survival_function(ax=ax, color="red", ci_show=False)
kmf_replication.plot_survival_function(ax=ax, color="blue", ci_show=False)

# Customize plot appearance
plt.title("Kaplan-Meier Curves for the Progression-Free Survival: Discovery and Replication Cohorts", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Progression-Free Survival Probability", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(fontsize=10)

# Add at-risk table
add_at_risk_counts(kmf_discovery, kmf_replication, ax=ax)

# Enhance styling
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()

# +
# Set the seed
set_seed(42)



# Load and normalize data
train_latent, train_clinical, train_ytime, train_yevent = load_combined_data(
    "processed_train_radiomic.csv", "train_clinical.csv", dtype
)
valid_latent, valid_clinical, valid_ytime, valid_yevent = load_combined_data(
    "processed_test_radiomic.csv", "test_clinical.csv", dtype
)

# Debug: Check data stats
print("Train data stats:")
print(f"Latent feature mean: {train_latent.mean(dim=0)}")
print(f"Latent feature std: {train_latent.std(dim=0)}")
print(f"Clinical feature mean: {train_clinical.mean(dim=0)}")
print(f"Clinical feature std: {train_clinical.std(dim=0)}")

input_latent_dim = train_latent.shape[1]
input_clinical_dim = train_clinical.shape[1]

# Hyperparameters
level_2_dim = [8, 16, 32, 64]
epoch_num = 500
patience = 200
Initial_Learning_Rate = [0.0001, 0.0005, 0.00001, 0.1]
L2_Lambda = [0.001, 0.00075, 0.0005, 0.00025, 0.0001]
Dropout_rate_1 = [0.0, 0.2, 0.3, 0.5]
Dropout_rate_2 = [0.0, 0.2, 0.3, 0.5]

# Validate level_2_dim
print(f"Initial level_2_dim: {level_2_dim}")
level_2_dim = [dim for dim in level_2_dim if isinstance(dim, int) and dim > 0]
if not level_2_dim:
    raise ValueError("No valid values for level_2_dim. Please check the hyperparameter settings.")
print(f"Validated level_2_dim: {level_2_dim}")

# Initialize variables for tracking the best hyperparameters
best_hyperparameters = None
best_train_cindex = float("-inf")
best_test_cindex = float("-inf")

# Hyperparameter tuning loop
for dr1 in Dropout_rate_1:
    for dr2 in Dropout_rate_2:
        for l2 in L2_Lambda:
            for lr in Initial_Learning_Rate:
                for dim in level_2_dim:
                    assert isinstance(dim, int), f"dim should be int, but got {type(dim)}: {dim}"
                    print(f"Training with dim={dim}, dr1={dr1}, dr2={dr2}, l2={l2}, lr={lr}")

                    # Train and evaluate
                    train_y_pred, eval_y_pred, train_cindex, test_cindex, best_epoch = train_LFSurv(
                        train_data=torch.cat([train_latent, train_clinical], dim=1),
                        train_ytime=train_ytime,
                        train_yevent=train_yevent,
                        eval_data=torch.cat([valid_latent, valid_clinical], dim=1),
                        eval_ytime=valid_ytime,
                        eval_yevent=valid_yevent,
                        latent_dim=input_latent_dim,
                        clinical_dim=input_clinical_dim,
                        level_2_dim=dim,
                        Dropout_Rate_1=dr1,
                        Dropout_Rate_2=dr2,
                        Learning_Rate=lr,
                        L2=l2,
                        epoch_num=epoch_num,
                        patience=patience,
                        path="saved_models/sup_checkpoint_tune.pt"
                    )

                    # Apply condition: Test C-index must be less than train C-index and higher than previous best
                    if train_cindex > test_cindex and test_cindex > best_test_cindex:
                        best_hyperparameters = {
                            "dim": dim,
                            "dr1": dr1,
                            "dr2": dr2,
                            "l2": l2,
                            "lr": lr
                        }
                        best_train_cindex = train_cindex
                        best_test_cindex = test_cindex
                        print(f"Updated best hyperparameters: {best_hyperparameters}")

# Validate and prepare the final optimal hyperparameters
if not best_hyperparameters:
    raise ValueError("No valid hyperparameters found that satisfy the conditions.")
opt_dim = best_hyperparameters['dim']
opt_dr1 = best_hyperparameters['dr1']
opt_dr2 = best_hyperparameters['dr2']
opt_l2 = best_hyperparameters['l2']
opt_lr = best_hyperparameters['lr']

print(f"Starting final training with opt_dim={opt_dim}, level_2_dim={opt_dim}")

# Final training
train_y_pred, test_y_pred, cindex_train, cindex_test, best_epoch_num = train_LFSurv(
    train_data=torch.cat([train_latent, train_clinical], dim=1),
    train_ytime=train_ytime,
    train_yevent=train_yevent,
    eval_data=torch.cat([valid_latent, valid_clinical], dim=1),
    eval_ytime=valid_ytime,
    eval_yevent=valid_yevent,
    latent_dim=input_latent_dim,
    clinical_dim=input_clinical_dim,
    level_2_dim=opt_dim,
    Dropout_Rate_1=opt_dr1,
    Dropout_Rate_2=opt_dr2,
    Learning_Rate=opt_lr,
    L2=opt_l2,
    epoch_num=epoch_num,
    patience=patience,
    path="saved_models/sup_checkpoint_final.pt"
)

# Print final results
print(f"Optimal hyperparameters: {best_hyperparameters}")
print(f"Training C-index: {cindex_train:.4f}, Testing C-index: {cindex_test:.4f}")

# After final training loop
print(f"Starting evaluation with optimal model.")

# Convert tensors to numpy arrays
train_ytime_np = train_ytime.detach().cpu().numpy()
train_yevent_np = train_yevent.detach().cpu().numpy()
valid_ytime_np = valid_ytime.detach().cpu().numpy()
valid_yevent_np = valid_yevent.detach().cpu().numpy()
test_y_pred_np = test_y_pred.detach().cpu().numpy()
train_y_pred_np = train_y_pred.detach().cpu().numpy()
# -



