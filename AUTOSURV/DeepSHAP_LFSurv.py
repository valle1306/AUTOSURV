# +
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



# +
import numpy as np
import pandas as pd
import torch
from LFSurv import LFSurv
import shap
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from lifelines.utils import concordance_index
import random

dtype = torch.float

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

### **Load Data (Latent Radiomic + Clinical)**
train_latent, train_clinical, train_ytime, train_yevent = load_combined_data(
    "processed_train_radiomic.csv", "train_clinical.csv", dtype
)
test_latent, test_clinical, test_ytime, test_yevent = load_combined_data(
    "processed_test_radiomic.csv", "test_clinical.csv", dtype
)

# **Concatenate Features for LFSurv Model**
train_data = torch.cat([train_latent, train_clinical], dim=1)
test_data = torch.cat([test_latent, test_clinical], dim=1)

# **Input Feature Dimensions**
input_latent_dim = train_latent.shape[1]
input_clinical_dim = train_clinical.shape[1]
input_dim = input_latent_dim + input_clinical_dim  # Total input dimensions for LFSurv

# **Load Pretrained LFSurv Model**
LFSurv_model = LFSurv(
    latent_dim=input_latent_dim,  
    input_n=input_latent_dim + input_clinical_dim,
    level_2_dim=32,  # Optimal found value
    Dropout_Rate_1=0.3,  # Optimal found value
    Dropout_Rate_2=0.3   # Optimal found value
)

LFSurv_model.load_state_dict(torch.load("saved_models/sup_checkpoint_final.pt", map_location=torch.device("cpu")))
LFSurv_model.eval()


### **Survival Analysis Metrics**
def calculate_harrell_cindex(y_time, y_event, y_pred):
    """Calculate Harrell's C-index for risk scores."""
    return concordance_index(y_time, -y_pred, y_event)

def calculate_unos_cindex(y_time, y_event, y_pred):
    """Calculate Uno's C-index for survival risk scores."""
    surv_data = Surv.from_arrays(event=y_event, time=y_time)
    return concordance_index_ipcw(surv_data, surv_data, -y_pred)[0]

# **Compute Prognostic Scores**
with torch.no_grad():
    train_scores = LFSurv_model(train_latent, train_clinical).detach().cpu().numpy().ravel()
    test_scores = LFSurv_model(test_latent, test_clinical).detach().cpu().numpy().ravel()


# **Compute C-Index Scores**
harrell_c_train = calculate_harrell_cindex(train_ytime.numpy(), train_yevent.numpy(), train_scores)
harrell_c_test = calculate_harrell_cindex(test_ytime.numpy(), test_yevent.numpy(), test_scores)

print(f"Harrell C-Index (Train): {harrell_c_train:.4f}")
print(f"Harrell C-Index (Test): {harrell_c_test:.4f}")

# **SHAP Explainer for LFSurv**
class PyTorchDeepExplainerLFSurv:
    def __init__(self, model, background_data):
        """
        SHAP Explainer for LFSurv survival model.
        - `model`: Pretrained LFSurv model.
        - `background_data`: Reference dataset for SHAP baseline.
        """
        self.model = model.eval()
        self.background_data = background_data

    def shap_values(self, data_to_explain):
        """
        Compute SHAP values for the input dataset.
        - Uses DeepExplainer to attribute feature importance to survival predictions.
        """
        explainer = shap.DeepExplainer(self.model, (self.background_data[0], self.background_data[1]))
        shap_values = shap_values = explainer.shap_values((data_to_explain[0], data_to_explain[1]))  # Use tuple
  # Ensure both latent & clinical are passed
        return shap_values


# **Define Background Data for SHAP (Subset of Training Data)**
background_data = train_data[:50]  # Use a subset for efficiency

# **Initialize SHAP Explainer**
explainer = PyTorchDeepExplainerLFSurv(LFSurv_model, background_data)

# **Explain Survival Risk Predictions**
test_shap_values = explainer.shap_values((test_latent, test_clinical))  # Pass both feature sets

# **Convert SHAP Values to DataFrame for Visualization**
shap_df = pd.DataFrame(test_shap_values, columns=[f"Feature_{i}" for i in range(input_dim)])

# **Save SHAP Values**
shap_df.to_csv("shap_values_LFSurv.csv", index=False)

# **Visualize SHAP Summary Plot**
shap.summary_plot(test_shap_values, test_data.numpy(), feature_names=[f"Feature_{i}" for i in range(input_dim)])

# +
import numpy as np
import pandas as pd
import torch
from LFSurv import LFSurv
import shap
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from lifelines.utils import concordance_index
import random
import shap

import tensorflow as tf
print(tf.__version__)


dtype = torch.float

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

### **Step 1: Load Model Checkpoint to Extract Expected Input Size**
checkpoint = torch.load("saved_models/sup_checkpoint_final.pt", map_location=torch.device("cpu"))

# Extract expected input size from checkpoint
expected_input_dim = checkpoint["c_fc1.weight"].shape[1]  # Extract total input feature size
expected_hidden_dim = checkpoint["c_fc1.weight"].shape[0]  # Extract hidden layer size

print(f"⚠️ Expected Model Input Dimension from Checkpoint: {expected_input_dim}")

### **Step 2: Load Data (Latent Radiomic + Clinical)**
train_latent, train_clinical, train_ytime, train_yevent = load_combined_data(
    "processed_train_radiomic.csv", "train_clinical.csv", dtype
)
test_latent, test_clinical, test_ytime, test_yevent = load_combined_data(
    "processed_test_radiomic.csv", "test_clinical.csv", dtype
)

### **Step 3: Fix Extra Column Issue in KL_PMVAE Processed Files**
# If the latent dimension is higher than expected, remove the extra index column
if train_latent.shape[1] > expected_input_dim - train_clinical.shape[1]:  
    print(f"⚠️ Removing extra column from latent features (KL_PMVAE output fix)")
    train_latent = train_latent[:, 1:]  # Remove first column
    test_latent = test_latent[:, 1:]

### **Step 4: Ensure Feature Dimensions Match**
input_latent_dim = train_latent.shape[1]
input_clinical_dim = train_clinical.shape[1]
input_dim = input_latent_dim + input_clinical_dim  # Total input dimensions for LFSurv

# Check if input dimensions match expected input size
if input_dim != expected_input_dim:
    raise ValueError(f"🚨 Mismatch: Model expects {expected_input_dim}, but received {input_dim}!")

### **Step 5: Load Pretrained LFSurv Model**
LFSurv_model = LFSurv(
    latent_dim=input_latent_dim,  
    input_n=input_dim,  # Ensure input_n matches saved model
    level_2_dim=expected_hidden_dim,  # Use hidden dimension from checkpoint
    Dropout_Rate_1=0.3,  # Optimal found value
    Dropout_Rate_2=0.3   # Optimal found value
)

# Load model weights
LFSurv_model.load_state_dict(checkpoint)
LFSurv_model.eval()

### **Step 6: Survival Analysis Metrics**
def calculate_harrell_cindex(y_time, y_event, y_pred):
    """Calculate Harrell's C-index for risk scores."""
    return concordance_index(y_time, -y_pred, y_event)

def calculate_unos_cindex(y_time, y_event, y_pred):
    """Calculate Uno's C-index for survival risk scores."""
    surv_data = Surv.from_arrays(event=y_event, time=y_time)
    return concordance_index_ipcw(surv_data, surv_data, -y_pred)[0]

# **Compute Prognostic Scores**
with torch.no_grad():
    train_scores = LFSurv_model(train_latent, train_clinical).detach().cpu().numpy().ravel()
    test_scores = LFSurv_model(test_latent, test_clinical).detach().cpu().numpy().ravel()

# **Compute C-Index Scores**
harrell_c_train = calculate_harrell_cindex(train_ytime.numpy(), train_yevent.numpy(), train_scores)
harrell_c_test = calculate_harrell_cindex(test_ytime.numpy(), test_yevent.numpy(), test_scores)

print(f"Harrell C-Index (Train): {harrell_c_train:.4f}")
print(f"Harrell C-Index (Test): {harrell_c_test:.4f}")


class PyTorchDeepExplainerLFSurv:
    def __init__(self, model, background_latent, background_clinical):
        """
        SHAP Explainer for LFSurv survival model.
        - `model`: Pretrained LFSurv model.
        - `background_latent`: Background latent features.
        - `background_clinical`: Background clinical features.
        """
        self.model = model.eval()  # Ensure model is in evaluation mode
        self.background_latent = background_latent
        self.background_clinical = background_clinical

        # **Fix: Ensure Model Wrapper is Not a Function**
        self.background_data = [background_latent, background_clinical]

        # ✅ **Pass the Model Directly to SHAP**
        self.explainer = shap.GradientExplainer(self.model, self.background_data)

    def shap_values(self, data_latent, data_clinical):
        """
        Compute SHAP values for the input dataset.
        Uses GradientExplainer for PyTorch compatibility.
        """
        input_data = [data_latent, data_clinical]  # Pass inputs as a list
        shap_values = self.explainer.shap_values(input_data)  
        return shap_values



# **Initialize SHAP Explainer**
# Define Background Data for SHAP (Subset of Training Data)
background_latent = train_latent[:50]  # Use a subset for efficiency
background_clinical = train_clinical[:50]

explainer = PyTorchDeepExplainerLFSurv(LFSurv_model, background_latent, background_clinical)

# **Explain Survival Risk Predictions**
test_shap_values = explainer.shap_values(test_latent, test_clinical)  # Pass both feature sets

# **Convert SHAP Values to DataFrame for Visualization**
shap_values_flat = np.squeeze(np.hstack(test_shap_values))  # Remove extra dimension
shap_df = pd.DataFrame(shap_values_flat.reshape(shap_values_flat.shape[0], -1), 
                       columns=[f"Feature_{i}" for i in range(input_dim)])

# **Save SHAP Values**
shap_df.to_csv("shap_values_LFSurv.csv", index=False)

# **Visualize SHAP Summary Plot**
test_input_data = np.hstack([test_latent.numpy(), test_clinical.numpy()])  # Stack both feature sets
shap.summary_plot(shap_values_flat, test_input_data, feature_names=[f"Feature_{i}" for i in range(input_dim)])




# -


