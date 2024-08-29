import torch
import torch.nn.functional as F
import numpy as np

def zeroth_order_loss(est, target, burn_in_size=0.0, align_mode='sampling', reduction='sum'):
    
    # Step 1: Slice features to match target
    if est.size(2) > target.size(2):
        est = est[:, :, :target.size(2)]

    # Determine the length of the series to use
    if align_mode == 'interpolation':
        # Step 2: Align time series by interpolating 'est' to match the length of 'target'
        est_aligned = torch.nn.functional.interpolate(est.transpose(1, 2), size=target.size(1), mode='linear', align_corners=False).transpose(1, 2)
    elif align_mode == 'sampling':
        # Step 2: Align time series by sampling
        # Calculate sampling rate based on the ratio of lengths
        if target.size(1) > est.size(1):
            step = target.size(1) // est.size(1)
            target_aligned = target[:, ::step, :]
            est_aligned = est
        else:
            step = est.size(1) // target.size(1)
            est_aligned = est[:, ::step, :]
            target_aligned = target
    else:
        raise ValueError("Invalid align_mode. Choose 'sampling' or 'interpolation'.")

    # Ensure both time series are of the same length after alignment
    min_length = min(est_aligned.size(1), target_aligned.size(1))
    est_aligned = est_aligned[:, :min_length, :]
    target_aligned = target_aligned[:, :min_length, :]

    # Step 3: Clip the initial portion of the time series based on burn_in_size
    burn_in_length = int(np.floor(burn_in_size * min_length))
    est_clipped = est_aligned[:, burn_in_length:, :]
    target_clipped = target_aligned[:, burn_in_length:, :]

    # Step 4: Calculate MSE Loss
    if reduction == 'sum':
        mse_loss = ((est_clipped - target_clipped) ** 2).sum(dim=-1).mean(dim=[0, 1])
    elif reduction == 'mean':
        mse_loss = ((est_clipped - target_clipped) ** 2).mean()
    else:
        raise ValueError('Reduction not recognized.')

    return mse_loss

class FirstOrderLoss(object):
    def __init__(self, dt_est=.01, dt_true=.01, der_weight=1e-4, burn_in_size=0.0):
        self.dt_est = dt_est
        self.dt_true = dt_true
        self.der_weight = der_weight
        self.burn_in_size = burn_in_size

    def __call__(self, est, target):

        zero_loss = zeroth_order_loss(est, target, burn_in_size=self.burn_in_size)

        der_est = torch.diff(est,dim=1) / self.dt_est
        der_target = torch.diff(target,dim=1) / self.dt_true
        one_loss = zeroth_order_loss(der_est, der_target, burn_in_size=self.burn_in_size)

        return zero_loss + self.der_weight * one_loss

class NthOrderLoss(object):
    def __init__(self, n=1, dt_est=.01, dt_true=.01, der_weight=1e-4, burn_in_size=0.0, fourier=False, fourier_weight=1e-4):
        self.dt_est = dt_est
        self.dt_true = dt_true
        self.der_weight = der_weight
        self.fourier = fourier
        self.fourier_weight = fourier_weight
        self.n = n
        self.burn_in_size = burn_in_size

    def __call__(self, est, target):

        total_loss = 0
        current_est = est
        current_target = target

        for i in range(self.n + 1):
            loss = zeroth_order_loss(current_est, current_target, burn_in_size=self.burn_in_size)
            total_loss += loss*(self.der_weight**i)

            current_est = torch.diff(current_est,dim=1) / self.dt_est**(i+1)
            current_target = torch.diff(current_target,dim=1) / self.dt_true**(i+1)

        if self.fourier:
            fft_est    = torch.abs(torch.fft.fft(est, dim=1))
            fft_target = torch.abs(torch.fft.fft(target, dim=1))
            loss = zeroth_order_loss(fft_est, fft_target, burn_in_size=0)
            total_loss += loss*self.fourier_weight

        return total_loss
