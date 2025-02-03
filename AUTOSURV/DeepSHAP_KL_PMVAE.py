# # Train again with input_n1 = 32
# * Since it is the optimal number of features that KL_PMVAE got reduced to

# +
import torch
import pandas as pd
from train_KL_PMVAE import train_KL_PMVAE
from KL_PMVAE import KL_PMVAE_Cancer

# Load processed train and test data
train_data = pd.read_csv("processed_train_radiomic.csv", index_col=0)
test_data = pd.read_csv("processed_test_radiomic.csv", index_col=0)

# Convert data to tensors
dtype = torch.FloatTensor
train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

# Define model parameters for retraining
input_n1 = 32  # Use reduced feature size
z_dim = 32  # Latent space dimension
level_2_dim = input_n1 // 2
learning_rate = 0.005
l2_regularization = 0.05
cutting_ratio = 0.3
num_epochs = 1000
num_cycles = 2

# Retrain the KL-PMVAE model
mean, logvar, val_mean, val_logvar, train_loss_unsup, eval_loss_unsup = train_KL_PMVAE(
    train_tensor,            # Training data
    test_tensor,             # Validation data
    z_dim,                   # Latent dimension
    input_n1,                # Reduced input size
    learning_rate,           # Learning rate
    l2_regularization,       # L2 regularization
    cutting_ratio,           # Cutting ratio for KL annealing
    num_epochs,              # Number of epochs
    num_cycles,              # Number of cycles
    dtype,                   # Data type
    save_model=True,         # Save the trained model
    path="saved_models/unsup_checkpoint_retrained.pt"  # Save path
)

print("Retraining complete.")

# +
import numpy as np
import pandas as pd
import torch
from KL_PMVAE import KL_PMVAE_Cancer
from utils import load_data
import json

# Load optimal parameters
with open("saved_models/optimal_params.json", "r") as f:
    optimal_params = json.load(f)

# Extract optimal parameters
opt_l2 = optimal_params["opt_l2"]
opt_lr = optimal_params["opt_lr"]
opt_dim = optimal_params["opt_dim"]
opt_epoch_num = optimal_params["opt_epoch_num"]
opt_num_cycle = optimal_params["opt_num_cycle"]
opt_cr = optimal_params["opt_cr"]

# Define dataset-specific parameters
input_n1 = 32  # Use reduced feature size
input_n2 = 0    # No clinical features
level_2_dim = input_n1 // 2  # Ensure consistency

# Load processed train and test data
train_data = pd.read_csv("processed_train_radiomic.csv", index_col=0)
test_data = pd.read_csv("processed_test_radiomic.csv", index_col=0)

# Load pre-trained KL_PMVAE model with optimal parameters
KL_PMVAE_model = KL_PMVAE_Cancer(
    z_dim=opt_dim,  # Optimal latent dimension
    input_n1=input_n1,  # Ensure consistency
    input_n2=input_n2,
    level_2_dim=level_2_dim  # Explicitly set level_2_dim
)

# Check model checkpoint for shape mismatches
checkpoint_path = "saved_models/unsup_checkpoint_retrained.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
print(f"Checkpoint encoder shape: {checkpoint['e_fc1.weight'].shape}")  # Should be (level_2_dim, input_n1)

# Load checkpoint
KL_PMVAE_model.load_state_dict(checkpoint)
KL_PMVAE_model.eval()

# Compute Prognostic Index (PI) for training set
with torch.no_grad():
    train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    print(f"Train tensor shape: {train_tensor.shape}")  # Debug shape
    mean, logvar, z, recon_x1 = KL_PMVAE_model(train_tensor)
    PI_train = mean.mean(dim=1).detach().cpu().numpy()

PI_med = np.median(PI_train)
print(f"Median Prognostic Index (PI_med): {PI_med:.4f}")

# Compute Prognostic Index (PI) for testing set
with torch.no_grad():
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    mean, logvar, z, recon_x1 = KL_PMVAE_model(test_tensor)
    PI_test = mean.mean(dim=1).detach().cpu().numpy()

# Divide testing set into high-risk and low-risk groups
high_risk_indices = np.where(PI_test > PI_med)[0]
low_risk_indices = np.where(PI_test <= PI_med)[0]

high_risk_data = test_data.iloc[high_risk_indices]
low_risk_data = test_data.iloc[low_risk_indices]

print(f"High-risk group size: {len(high_risk_indices)}, Low-risk group size: {len(low_risk_indices)}")

# SHAP explainer for high-risk and low-risk groups
class PyTorchDeepExplainerRadiomic:
    def __init__(self, model, data, output_number, latent_dim, explain_latent_space):
        self.model = model.eval()
        self.data = data
        self.output_number = output_number
        self.latent_dim = latent_dim
        self.explain_latent_space = explain_latent_space

    def shap_values(self, data_to_explain):
        # Implement SHAP explanation logic
        pass

# Initialize and use the SHAP explainer
high_risk_tensor = torch.tensor(high_risk_data.values, dtype=torch.float32)
low_risk_tensor = torch.tensor(low_risk_data.values, dtype=torch.float32)

explainer = PyTorchDeepExplainerRadiomic(
    KL_PMVAE_model,
    train_tensor,
    output_number=0,
    latent_dim=opt_dim,
    explain_latent_space=True
)

# Explain high-risk group
high_risk_shap_values = explainer.shap_values(high_risk_tensor)

# Explain low-risk group
low_risk_shap_values = explainer.shap_values(low_risk_tensor)



# +
import matplotlib.pyplot as plt

plt.hist(PI_train, bins=20)
plt.xlabel("PI Values")
plt.ylabel("Frequency")
plt.title("Distribution of Prognostic Index (PI)")
plt.show()

print("PI_train min:", np.min(PI_train))
print("PI_train max:", np.max(PI_train))
print("PI_train mean:", np.mean(PI_train))
print("PI_train median:", np.median(PI_train))

# -

print("Mean latent representation (train):", np.mean(mean.detach().cpu().numpy(), axis=0))
print("Standard deviation:", np.std(mean.detach().cpu().numpy(), axis=0))


# Observations:
#
# * PI Vvalues are near-zero, meaning that all prognostic index values are extremely small and do not vary across different samples. 
# * Many of the mean latent values are close to 0 --> sign of posterior collapse?
# * Since the variance across latent space dimensions is also very small, the latent fature are not encoding meaningful differences between samples
#
# Causes: many possible causes, including:
# * KL regularization is too strong: KL loss terms is weighted too heavily, the model forces the latent distribution too close to 0
# * If input data has been scaled properly, the model may learn an amost constant representation
# * z_dim might be too small; could increase z_dim


