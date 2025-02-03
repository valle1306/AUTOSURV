#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import math
import torch
import torch.nn as nn
from torch.autograd import Variable

dtype = torch.FloatTensor


# %%


class KL_PMVAE_Cancer(nn.Module):
    def __init__(self, z_dim, input_n1, input_n2, level_2_dim=None):
        super(KL_PMVAE_Cancer, self).__init__()

        self.input_n1 = input_n1  # Radiomic features
        self.input_n2 = input_n2  # Clinical features
        self.level_2_dim = level_2_dim if level_2_dim is not None else input_n1 // 2
        self.z_dim = z_dim  # Latent space dimension

        # Activation functions
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

        # ENCODER layers
        self.e_fc1 = nn.Linear(self.input_n1, self.level_2_dim)
        self.e_bn1 = nn.BatchNorm1d(self.level_2_dim)
        
        # Adjust Level 2 layers if x2 is not provided
        self.e_fc2_mean = nn.Linear(self.level_2_dim + (self.input_n2 if self.input_n2 > 0 else 0), self.z_dim)
        self.e_fc2_logvar = nn.Linear(self.level_2_dim + (self.input_n2 if self.input_n2 > 0 else 0), self.z_dim)
        self.e_bn2_mean = nn.BatchNorm1d(self.z_dim)
        self.e_bn2_logvar = nn.BatchNorm1d(self.z_dim)

        # DECODER layers
        self.d_fc2 = nn.Linear(self.z_dim, self.level_2_dim)
        self.d_bn2 = nn.BatchNorm1d(self.level_2_dim)
        self.d_fc1 = nn.Linear(self.level_2_dim, self.input_n1)
        self.d_bn1 = nn.BatchNorm1d(self.input_n1)

        # Dropout layers
        self.dropout_1 = nn.Dropout(0.)
        self.dropout_2 = nn.Dropout(0.)

    def _reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn_like(std)
        return z.mul(std) + mean

    def encode(self, x1, x2=None, s_dropout=False):
        if s_dropout:
            x1 = self.dropout_1(x1)
        level2_layer = self.relu(self.e_bn1(self.e_fc1(x1)))

        if x2 is not None:
            conc_layer = torch.cat((level2_layer, x2), 1)
        else:
            conc_layer = level2_layer

        if s_dropout:
            conc_layer = self.dropout_2(conc_layer)

        latent_mean = self.e_bn2_mean(self.e_fc2_mean(conc_layer))
        latent_logvar = self.e_bn2_logvar(self.e_fc2_logvar(conc_layer))
        return latent_mean, latent_logvar

    def decode(self, z):
        level2_layer = self.relu(self.d_bn2(self.d_fc2(z)))
        recon_x1 = self.sigm(self.d_bn1(self.d_fc1(level2_layer)))
        return recon_x1

    def forward(self, x1, x2=None, s_dropout=False):
        mean, logvar = self.encode(x1, x2, s_dropout)
        z = self._reparameterize(mean, logvar)
        recon_x1 = self.decode(z)
        return mean, logvar, z, recon_x1



# %%


class KL_PMVAE_genes(nn.Module):

    def __init__(self, z_dim, input_n, Pathway_Mask):
        super(KL_PMVAE_genes, self).__init__()
        self.pathway_mask = Pathway_Mask
        level_2_dim = Pathway_Mask.size(0)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.z_dim = z_dim
        # ENCODER fc layers
        # level 1
        self.e_fc1 = nn.Linear(input_n, level_2_dim)
        self.e_bn1 = nn.BatchNorm1d(level_2_dim)
        
        # level 2
        self.e_fc2_mean = nn.Linear(level_2_dim, z_dim)
        self.e_fc2_logvar = nn.Linear(level_2_dim, z_dim)
        self.e_bn2_mean = nn.BatchNorm1d(z_dim)
        self.e_bn2_logvar = nn.BatchNorm1d(z_dim)
        
        # DECODER fc layers
        # level 2
        self.d_fc2 = nn.Linear(z_dim, level_2_dim)
        self.d_bn2 = nn.BatchNorm1d(level_2_dim)
        
        # level 1
        self.d_fc1 = nn.Linear(level_2_dim, input_n)
        self.d_bn1 = nn.BatchNorm1d(input_n)
        
        # Dropout
        self.dropout_1 = nn.Dropout(0.)
        self.dropout_2 = nn.Dropout(0.)
    
    def _reparameterize(self, mean, logvar, z = None):
        std = logvar.mul(0.5).exp()    
        if z is None:
            if torch.cuda.is_available():
                z = Variable(torch.cuda.FloatTensor(std.size()).normal_(0, 1))
            else:
                z = Variable(torch.FloatTensor(std.size()).normal_(0, 1))
        return z.mul(std) + mean
    
    def encode(self, x1, s_dropout):
        if s_dropout:
            x1 = self.dropout_1(x1)
        self.e_fc1.weight.data = self.e_fc1.weight.data.mul(self.pathway_mask)
        level2_layer = self.relu(self.e_bn1(self.e_fc1(x1)))
        
        if s_dropout:
            level2_layer = self.dropout_2(level2_layer)
        latent_mean = self.e_bn2_mean(self.e_fc2_mean(level2_layer))
        latent_logvar = self.e_bn2_logvar(self.e_fc2_logvar(level2_layer))
        
        return latent_mean, latent_logvar
    
    def decode(self, z):
        level2_layer = self.relu(self.d_bn2(self.d_fc2(z)))
        
        self.d_fc1.weight.data = self.d_fc1.weight.data.mul(torch.transpose(self.pathway_mask, 0, 1))
        
        recon_x = self.sigm(self.d_bn1(self.d_fc1(level2_layer)))
        
        return recon_x
        
    def forward(self, x1, s_dropout = False):
        mean, logvar = self.encode(x1, s_dropout)
        z = self._reparameterize(mean, logvar)
        recon_x = self.decode(z)
                    
        return mean, logvar, z, recon_x


# %%


class KL_PMVAE_mirnas(nn.Module):

    def __init__(self, z_dim, input_n):
        super(KL_PMVAE_mirnas, self).__init__()
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.z_dim = z_dim
        
        # ENCODER fc layers
        # level 1
        self.e_fc1_mean = nn.Linear(input_n, z_dim)
        self.e_fc1_logvar = nn.Linear(input_n, z_dim)
        self.e_bn1_mean = nn.BatchNorm1d(z_dim)
        self.e_bn1_logvar = nn.BatchNorm1d(z_dim)
        
        # DECODER fc layers
        # level 1
        self.d_fc1 = nn.Linear(z_dim, input_n)
        self.d_bn1 = nn.BatchNorm1d(input_n)
        
        # Dropout
        self.dropout_1 = nn.Dropout(0.)
    
    def _reparameterize(self, mean, logvar, z = None):
        std = logvar.mul(0.5).exp()    
        if z is None:
            if torch.cuda.is_available():
                z = Variable(torch.cuda.FloatTensor(std.size()).normal_(0, 1))
            else:
                z = Variable(torch.FloatTensor(std.size()).normal_(0, 1))
        return z.mul(std) + mean
    
    def encode(self, x1, s_dropout):
        if s_dropout:
            x1 = self.dropout_1(x1)
        latent_mean = self.e_bn1_mean(self.e_fc1_mean(x1))
        latent_logvar = self.e_bn1_logvar(self.e_fc1_logvar(x1))
        
        return latent_mean, latent_logvar
    
    def decode(self, z):
        recon_x = self.sigm(self.d_bn1(self.d_fc1(z)))
        
        return recon_x
        
    def forward(self, x1, s_dropout = False):
        mean, logvar = self.encode(x1, s_dropout)
        z = self._reparameterize(mean, logvar)
        recon_x = self.decode(z)
                    
        return mean, logvar, z, recon_x


# %%




