import torch

def train_one_epoch(model, data_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    for batch in data_loader:
        # Assume batch returns x_filled and mask
        x_filled = batch['data'].to(device)
        mask = batch['mask'].to(device)

        y_pred = model(x_filled)

        loss = loss_fn(y_pred, x_filled, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(data_loader)



@torch.no_grad()  # Ensures no gradients are stored
def validate(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0
    for batch in val_loader:
        x_filled = batch['data'].to(device)
        mask = batch['mask'].to(device)

        y_pred = model(x_filled)
        loss = loss_fn(y_pred, x_filled, mask)

        total_loss += loss.item()

    return total_loss / len(val_loader)