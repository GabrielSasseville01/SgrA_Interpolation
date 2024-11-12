import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import layers
import pdb

class Gaussian:
    mean = None
    logvar = None

class LossInfo:
    loglik = None
    mse = None
    mae = None
    composite_loss = None


class TRIPLETFORMER(nn.Module):

    def __init__(
        self,
        input_dim=41,
        enc_num_heads=4,
        dec_num_heads=4,
        num_ref_points=128,
        mse_weight=1.,
        norm=True,
        imab_dim = 128,
        cab_dim = 128,
        decoder_dim = 128,
        n_layers=2,
        device='cuda'):
        super().__init__()
        self.dim=input_dim
        self.enc_num_heads=enc_num_heads
        self.dec_num_heads=dec_num_heads
        self.num_ref_points=num_ref_points
        self.mse_weight=mse_weight
        self.norm=norm
        self.imab_dim = imab_dim
        self.cab_dim = cab_dim
        self.decoder_dim = decoder_dim
        self.n_layers=n_layers
        self.device=device
        self.enc = layers.Encoder(self.dim, self.imab_dim, self.n_layers, self.num_ref_points, self.enc_num_heads, device=device)
        self.dec_att = layers.Decoder_att(self.dim, self.imab_dim, self.cab_dim, self.dec_num_heads, device=device)
        self.O = layers.output(self.cab_dim, self.decoder_dim, device=device)

    def encode(self, context_x, context_w, target_x):
        mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        Z_e, mk_e = self.enc(context_x, X, mask)
        return Z_e, mk_e

    def decode(self, Z_e, mk_e, target_context, target_mask):
        px = Gaussian()

        Z_d = self.dec_att(Z_e, mk_e, target_context, target_mask)
        dec_out = self.O(Z_d)
        px.mean = dec_out[:,:,0:1]
        px.logvar = torch.log(1e-8 + F.softplus(dec_out[:, :, 1:]))
        return px

    def get_interpolation(self, context_x, context_y, target_x, target_context, target_mask):
        Z_e, mk_e = self.encode(context_x, context_y, target_x)
        px = self.decode(Z_e, mk_e, target_context, target_mask)
        return px

    def compute_loglik(self, target_y, px, norm=True):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        log_p = utils.log_normal_pdf(
            target, px.mean, px.logvar, mask).sum(-1).sum(-1)
        if norm:
            return log_p / mask.sum(-1).sum(-1)
        return log_p

    def compute_mse(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_squared_error(target, pred, mask)

    def compute_mae(self, target_y, pred):
        target, mask = target_y[:, :, :self.dim2], target_y[:, :, self.dim2:]
        return utils.mean_absolute_error(target, pred, mask)


    def compute_unsupervised_loss(
        self, context_x, context_y, target_x, target_y, num_samples=1, beta=1.
    ):
        loss_info = LossInfo()

        tau = target_x[:,:,None].repeat(1,1,self.dim) # Time indicator
        U = target_y[:,:,:self.dim] # Ground truths for values to be predicted
        mk = target_y[:,:,self.dim:] # Reconstruction mask. 1's correspond to values to be predicted.
        C = torch.ones(mk.size(), dtype=torch.int64).cumsum(-1) - 1 
        C = C.to(self.device) # Channel index indicator. (Batch, t, dim) where each tensor is [0, 1, 2, 3]
        mk_bool = mk.to(torch.bool) # Reconstruction mask as boolean values

        full_len = tau.size(1)*self.dim # Timesteps * dim. 960*4
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0) # Padding function. Pads with 0's on the right until full_len

        # Keep only values to be reconstructed. Pad with 0's. (Batch, t, dim) --> (Batch, full_len)
        # Iterate over batches. r = tau/U/mk/C for given batch. m = mask for given batch.
        # r[m] = values of r where m is True flattened into one dimension.
        # pad(r[m]) = r[m] padded with zeros until full_len
        # Final tensors should have shape (Batch, full_len) 
        tau = torch.stack([pad(r[m]) for r, m in zip(tau, mk_bool)]).contiguous()
        U = torch.stack([pad(r[m]) for r, m in zip(U, mk_bool)]).contiguous()
        mk = torch.stack([pad(r[m]) for r, m in zip(mk, mk_bool)]).contiguous()
        C = torch.stack([pad(r[m]) for r, m in zip(C, mk_bool)]).contiguous()
        C_ = C # Channel indicators for each reconstructed value

        C = torch.nn.functional.one_hot(C, num_classes =self.dim)

        # Here is an example of the above code
        # Original C is after line 97:
        # tensor([[[0, 1, 2, 3],
        #         [0, 1, 2, 3]]])
        # mk_bool is:
        # tensor([[[ True, False, False,  True],
        #         [False,  True,  True, False]]])
        # C after Padding is (also corresponds to C_):
        # tensor([[0, 3, 1, 2, 0, 0, 0, 0]])
        # C after one-hot is:
        # tensor([[[1, 0, 0, 0],
        #         [0, 0, 0, 1],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [1, 0, 0, 0],
        #         [1, 0, 0, 0],
        #         [1, 0, 0, 0],
        #         [1, 0, 0, 0]]])
        
        target_context = torch.cat([tau[:,:,None], C], -1).contiguous() # Adds time indicator to first element of each one-hot encoded vector
        target_mask = torch.stack([C_, mk], -1)

        # In the previous example, target_mask would be:
        # tensor(
        # [[[0., 1.],
        #  [3., 1.],
        #  [1., 1.],
        #  [2., 1.],
        #  [0., 0.],
        #  [0., 0.],
        #  [0., 0.],
        #  [0., 0.]]])
        # And obs_len would be 4
    
        
        obs_len = torch.max(target_mask[:,:,1].sum(-1)).to(torch.int64) # number of points to predict
        # Truncate the padding to the number of points to predict
        target_context = target_context[:, :obs_len]
        target_mask = target_mask[:, :obs_len]
        target_vals = U[:,:obs_len]

        mask = torch.cat([target_vals[:,:,None], target_mask[:,:,1:]],-1)

        px = self.get_interpolation(context_x, context_y, target_x, target_context, target_mask)
    
        self.dim2 = 1

        loglik = self.compute_loglik(mask, px, self.norm)
        loss_info.loglik = loglik.mean() # log-likelihood
        loss_info.mse = self.compute_mse(mask, px.mean)
        loss_info.mae = self.compute_mae(mask, px.mean)
        loss_info.composite_loss = -loss_info.loglik + self.mse_weight * loss_info.mse # NLL + MSE
        return loss_info
    

    def inference(
        self, context_x, context_y, target_x, target_y, num_samples=1, beta=1.
    ):
        tau = target_x[:,:,None].repeat(1,1,self.dim) # Time indicator
        time = torch.arange(tau.size(1), device=self.device).unsqueeze(-1).repeat(1, self.dim).unsqueeze(0).repeat(tau.size(0), 1, 1) 
        U = target_y[:,:,:self.dim] # Ground truths for values to be predicted
        mk = target_y[:,:,self.dim:] # Reconstruction mask. 1's correspond to values to be predicted.
        C = torch.ones(mk.size(), dtype=torch.int64).cumsum(-1) - 1 
        C = C.to(self.device) # Channel index indicator. (Batch, t, dim) where each tensor is [0, 1, 2, 3]
        mk_bool = mk.to(torch.bool) # Reconstruction mask as boolean values

        full_len = tau.size(1)*self.dim # Timesteps * dim. 960*4
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0) # Padding function. Pads with 0's on the right until full_len

        # Keep only values to be reconstructed. Pad with 0's. (Batch, t, dim) --> (Batch, full_len)
        # Iterate over batches. r = tau/U/mk/C for given batch. m = mask for given batch.
        # r[m] = values of r where m is True flattened into one dimension.
        # pad(r[m]) = r[m] padded with zeros until full_len
        # Final tensors should have shape (Batch, full_len) 
        tau = torch.stack([pad(r[m]) for r, m in zip(tau, mk_bool)]).contiguous()
        time = torch.stack([pad(r[m]) for r, m in zip(time, mk_bool)]).contiguous()
        U = torch.stack([pad(r[m]) for r, m in zip(U, mk_bool)]).contiguous()
        mk = torch.stack([pad(r[m]) for r, m in zip(mk, mk_bool)]).contiguous()
        C = torch.stack([pad(r[m]) for r, m in zip(C, mk_bool)]).contiguous()
        C_ = C # Channel indicators for each reconstructed value

        C = torch.nn.functional.one_hot(C, num_classes =self.dim)

        # Here is an example of the above code
        # Original C is after line 97:
        # tensor([[[0, 1, 2, 3],
        #         [0, 1, 2, 3]]])
        # mk_bool is:
        # tensor([[[ True, False, False,  True],
        #         [False,  True,  True, False]]])
        # C after Padding is (also corresponds to C_):
        # tensor([[0, 3, 1, 2, 0, 0, 0, 0]])
        # C after one-hot is:
        # tensor([[[1, 0, 0, 0],
        #         [0, 0, 0, 1],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [1, 0, 0, 0],
        #         [1, 0, 0, 0],
        #         [1, 0, 0, 0],
        #         [1, 0, 0, 0]]])
        
        target_context = torch.cat([tau[:,:,None], C], -1).contiguous() # Adds time indicator to first element of each one-hot encoded vector
        target_mask = torch.stack([C_, mk], -1)

        # In the previous example, target_mask would be:
        # tensor(
        # [[[0., 1.],
        #  [3., 1.],
        #  [1., 1.],
        #  [2., 1.],
        #  [0., 0.],
        #  [0., 0.],
        #  [0., 0.],
        #  [0., 0.]]])
        # And obs_len would be 4
    
        
        obs_len = torch.max(target_mask[:,:,1].sum(-1)).to(torch.int64) # number of points to predict
        print('Number of points to predict: ', obs_len)
        # Truncate the padding to the number of points to predict
        target_context = target_context[:, :obs_len]
        target_mask = target_mask[:, :obs_len]
        time_indices = time[:, :obs_len]
        channel_indices = target_mask[:, :, 0]  # Shape: (Batch, obs_len)

        px = self.get_interpolation(context_x, context_y, target_x, target_context, target_mask)

        return px, time_indices, channel_indices


