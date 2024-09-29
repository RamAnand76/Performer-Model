import torch
import time

def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    start_time = time.time()  # Record the start time
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Record the start time for this epoch
        epoch_loss = 0
        
        for i, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            
            # Create masks
            src_mask = None
            tgt_mask = torch.triu(torch.ones(tgt_input.size(1), tgt_input.size(1)), diagonal=1).bool().to(device)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate time taken so far
            elapsed_time = time.time() - start_time
            avg_epoch_time = (time.time() - epoch_start_time) / (i + 1)
            remaining_epochs = num_epochs - epoch - 1
            avg_epoch_time_all = elapsed_time / (epoch + 1)
            remaining_time = avg_epoch_time_all * remaining_epochs

            print(f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            print(f'Time taken so far: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
            print(f'Estimated time to finish: {remaining_time // 60:.0f}m {remaining_time % 60:.0f}s')
        
        # Print the average loss for the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {epoch_loss / len(dataloader):.4f}')
        print(f'Time for epoch {epoch + 1}: {time.time() - epoch_start_time:.0f}s')

    total_time = time.time() - start_time
    print(f'Training completed in: {total_time // 60:.0f}m {total_time % 60:.0f}s')
