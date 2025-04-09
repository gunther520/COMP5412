import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
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
        collate_fn=variable_length_collate if fixed_length is None else None
    )
    
    # Create validation loader for rank 0
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            collate_fn=variable_length_collate if fixed_length is None else None
        )
    

    # Move model to the device
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    
    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )
    
    # Loss and optimizer
    criterion = criterion
    optimizer = optimizer
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for batch in train_loader:
            noisy = batch['noisy'].unsqueeze(1).to(device)
            clean = batch['clean'].unsqueeze(1).to(device)
            
            # Forward pass
            output = model(noisy)

            
            # Compute loss
            loss = criterion(output, clean)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Validation step (only on rank 0)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_noisy = val_batch['noisy'].unsqueeze(1).to(device)
                    val_clean = val_batch['clean'].unsqueeze(1).to(device)
                    
                    # Forward pass
                    val_output = model(val_noisy)

                    
                    # Compute loss
                    loss = criterion(val_output, val_clean)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), "../checkpoints/best_model_ddp.pth")
    
    # Cleanup
    torch.distributed.destroy_process_group()
    print(f"Rank {rank} training complete")

def run_ddp_training(model,criterion, optimizer,train_dataset, val_dataset, batch_size, fixed_length, 
                    variable_length_collate, num_epochs=10):
    # Get world size
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