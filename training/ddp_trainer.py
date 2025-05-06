import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def setup_ddp(rank, world_size):
    """Setup for distributed data parallel"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU training
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def train_ddp(rank, model, criterion, optimizer,world_size, train_dataset, val_dataset, 
              batch_size, fixed_length, variable_length_collate, num_epochs=10):
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # Create dataloaders with the distributed sampler
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=variable_length_collate if fixed_length is None else None,
        num_workers=24,
        pin_memory=True
    )
    
    # Create validation loader for rank 0
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            collate_fn=variable_length_collate if fixed_length is None else None,
            num_workers=24,
            pin_memory=True
        )
    

    # Move model to the device
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)

        # Convert BatchNorm1d to SyncBatchNorm HERE (after DDP setup)
    if any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules()):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    scaler = GradScaler()

    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank,find_unused_parameters=False
    )
    
    # Loss and optimizer

    optimizer = optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',     # Reduce LR when the metric stops decreasing
        factor=0.2,     # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5,     # Number of epochs with no improvement after which learning rate will be reduced.
        min_lr=1e-7     # Lower bound on the learning rate
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for i,batch in enumerate(train_loader):
            noisy = batch['noisy'].unsqueeze(1).to(device,non_blocking=True)
            clean = batch['clean'].unsqueeze(1).to(device,non_blocking=True)
            noise = batch['noise'].unsqueeze(1).to(device,non_blocking=True)
            #print(f"Input range: {noisy.min().item():.3f} to {noisy.max().item():.3f}")
            #print(f"Target range: {clean.min().item():.3f} to {clean.max().item():.3f}")
            mask = batch['mask'].unsqueeze(1).to(device, non_blocking=True)
            # Forward pass

            optimizer.zero_grad()
            loss=0
            with autocast("cuda"):
                output = model(noisy)
                # Calculate MSE loss with masking (weight: 1.0)
                #element_mse = criterion[0](output, clean)  # This returns per-element losses
                #masked_mse = (element_mse * mask).sum() / (mask.sum() + 1e-8)
                output_clean, output_noise = output["speech"], output["noise"]
                # Calculate L1 loss with masking (weight: 1.0)
                element_l1 = criterion[1](output_clean, clean) 
                noise_l1 = criterion[1](output_noise, noise)

                masked_l1 = ((element_l1 * mask).sum()+ (noise_l1*mask).sum()) / (mask.sum() + 1e-8)
                
                # Combine losses with weights (e.g., 0.5 for MSE, 0.5 for L1)
                loss = masked_l1
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Adjust 
            scaler.step(optimizer)
            
            scaler.update()

            
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Validation step (only on rank 0)
            model.eval()
            val_loss = 0.0
            with torch.no_grad(),autocast("cuda"):
                for val_batch in val_loader:
                    val_noisy = val_batch['noisy'].unsqueeze(1).to(device,non_blocking=True)
                    val_clean = val_batch['clean'].unsqueeze(1).to(device, non_blocking=True)
                    val_noise = val_batch['noise'].unsqueeze(1).to(device, non_blocking=True)
                    val_mask = val_batch['mask'].unsqueeze(1).to(device, non_blocking=True)
    
                    # Forward pass
                    val_output = model(val_noisy)
                    val_output_clean,val_output_noise = val_output["speech"], val_output["noise"]

                    # Calculate masked losses
                    #element_mse = criterion[0](val_output, val_clean)
                    #masked_mse = (element_mse * val_mask).sum() / (val_mask.sum() + 1e-8)
                    
                    element_l1 = criterion[1](val_output_clean, val_clean)  # This returns per-element losses
                    noise_l1 = criterion[1](val_output_noise, val_noise)  # This returns per-element losses
                    # Calculate L1 loss with masking (weight: 1.0)

                    masked_l1 = ((element_l1 * val_mask).sum() + (noise_l1 * val_mask).sum()) / (val_mask.sum() + 1e-8)
                    
                    # Combine losses with weights
                    val_batch_loss = masked_l1
                    val_loss += val_batch_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), "../checkpoints/best_model_ddp.pth")

            scheduler.step(avg_val_loss)
            print(f"Current LR: {scheduler.get_last_lr()[0]:.7f}")
    
    # Cleanup
    torch.distributed.destroy_process_group()
    print(f"Rank {rank} training complete")

def run_ddp_training(model,criterion, optimizer,train_dataset, val_dataset, batch_size, fixed_length, 
                    variable_length_collate, num_epochs=10):
    # Get world size
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            train_ddp,
            args=(model,criterion, optimizer,world_size,train_dataset, val_dataset, 
                 batch_size, fixed_length, variable_length_collate, num_epochs),
            nprocs=world_size,
            join=True
        )
        return "best_model_ddp.pth"
    else:
        print("Only one GPU available. DDP not used.")

        return None