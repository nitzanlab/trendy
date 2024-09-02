import time
import torch

def process_batch(batch, model, use_log_scale=False, scheduler=None, device='cpu', clip_target=-1):

    # Get time series and parameters from batch
    target = batch['X'].to(device)[:,:clip_target]
    init   = target[:,0]
    params = batch['p'].to(device)

    model_core = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Preprocess target if necessary
    if use_log_scale:
        target = torch.log10(target)

    # If target time series has not been processed with pca, and pca is part of the pipeline
    if target.shape[-1] != model_core.node_input_dim and hasattr(model_core, 'pca_layer'):
        b, t, f = target.shape
        target = target.reshape(-1,f)
        target = model_core.pca_layer(target).reshape(b,t,-1)
    else:
        raise ValueError('There is a shape mismatch between target and estimate.')

    # Clip target according to scheduler
    if scheduler is not None:
        target_samples = int(scheduler.current_prop * target.shape[1])
        target = target[:,:target_samples]

    # Run Trendy
    est = model_core.run(init, params)

    return target, est

def run_epoch(dl, model, criterion, optimizer, scheduler, train=True, use_log_scale=False, device='cpu', clip_target=-1):
    total_loss = 0
    model.train() if train else model.eval()

    batch_size = dl.batch_size
    io_time = 0
    update_time = 0

    with torch.set_grad_enabled(train):
        io_start = time.time()
        for i, data in enumerate(dl, 0):
            io_stop = time.time()

            optimizer.zero_grad()

            target, est = process_batch(data, model, use_log_scale=use_log_scale, scheduler=scheduler, device=device, clip_target=clip_target)

            # Loss
            loss = criterion(est, target).mean()

            if train:
                loss.backward()
                optimizer.step()
            update_stop = time.time()

            total_loss  += loss.item()
            io_time += (io_stop - io_start)
            update_time += (update_stop - io_stop)
            io_start = time.time()
    print(f'Epoch completed. io time: {io_time / (i + 1):.4f} s. update time: {update_time / (i + 1):.4f} s')
               
    return total_loss / len(dl)
