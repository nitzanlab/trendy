from trendy.train import FirstOrderLoss
import torch

criterion = FirstOrderLoss(dt_est=.01, dt_true=.01, der_weight=1e-4, burn_in_size=0.5)

est = torch.rand(32, 100, 16)
target = est

print(f'Loss : {criterion(est, target)}')
