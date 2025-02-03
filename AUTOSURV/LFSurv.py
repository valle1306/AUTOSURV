# +
import torch
import torch.nn as nn

class LFSurv(nn.Module):
    def __init__(self, latent_dim, input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2):
        super(LFSurv, self).__init__()
        self.latent_dim = latent_dim
        self.input_n = input_n  # Total input dimension (latent_dim + clinical_dim)
        self.level_2_dim = level_2_dim
        self.tanh = nn.Tanh()

        # Batch normalization for latent features
        self.c_bn_input = nn.BatchNorm1d(latent_dim)
        print(f"input_n: {self.input_n}, level_2_dim: {self.level_2_dim}")
        print(f"Expected Linear layer shape: {self.level_2_dim} x {self.input_n}")

        # Fully connected layers
        self.c_fc1 = nn.Linear(self.input_n, self.level_2_dim)
        self.c_bn2 = nn.BatchNorm1d(self.level_2_dim)
        self.c_fc2 = nn.Linear(self.level_2_dim, 1, bias=False)
        self.c_fc2.weight.data.uniform_(-0.001, 0.001)

        # Dropout layers
        self.dropout_1 = nn.Dropout(Dropout_Rate_1)
        self.dropout_2 = nn.Dropout(Dropout_Rate_2)

    def coxlayer(self, latent_features, clinical_features, s_dropout):
        # Debug latent features
        #print(f"Latent features shape before dropout: {latent_features.shape}")
        if s_dropout:
            latent_features = self.dropout_1(latent_features)

        # Apply batch normalization to latent features
        #print(f"Latent features shape before BatchNorm: {latent_features.shape}")
        latent_features = self.c_bn_input(latent_features)

        # Handle clinical features dimension mismatch (padding if necessary)
        if clinical_features.size(1) < self.input_n - self.latent_dim:
            padding = self.input_n - self.latent_dim - clinical_features.size(1)
            clinical_features = torch.nn.functional.pad(clinical_features, (0, padding))
            #print(f"Clinical features padded to shape: {clinical_features.shape}")

        # Concatenate latent and clinical features
        combined_features = torch.cat((latent_features, clinical_features), dim=1)
        #print(f"Combined features shape: {combined_features.shape}")

        # Fully connected layers
        hidden_layer = self.tanh(self.c_fc1(combined_features))
        #print(f"Hidden layer shape after fc1 and tanh: {hidden_layer.shape}")
        if s_dropout:
            hidden_layer = self.dropout_2(hidden_layer)
            #print(f"Hidden layer shape after dropout: {hidden_layer.shape}")
        hidden_layer = self.c_bn2(hidden_layer)
        #print(f"Hidden layer shape after BatchNorm: {hidden_layer.shape}")

        # Final linear layer for predictions
        y_pred = self.c_fc2(hidden_layer)
        #print(f"Final output shape (y_pred): {y_pred.shape}")
        return y_pred

    def forward(self, latent_features, clinical_features, s_dropout=False):
        """
        Forward pass through the network.
        """
        return self.coxlayer(latent_features, clinical_features, s_dropout)



# -


