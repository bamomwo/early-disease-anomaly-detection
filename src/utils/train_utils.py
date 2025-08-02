import torch
import numpy as np

def train_one_epoch(model, train_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Assume batch returns x_filled and mask
        x_filled = batch['data'].to(device)
        mask = batch['mask'].to(device)

        y_pred = model(x_filled)

        loss = loss_fn(y_pred, x_filled, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(train_loader)



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


@torch.no_grad()
def evaluate(model, test_loader, device, loss_fn):
    model.eval()
    all_losses = []
    all_inputs = []
    all_outputs = []
    all_masks  = []              # ← new

    for batch in test_loader:
        x_filled = batch['data'].to(device)
        mask     = batch['mask'].to(device)

        y_pred = model(x_filled)
        loss   = loss_fn(y_pred, x_filled, mask)

        all_losses.append(loss.item())
        all_inputs.append(x_filled.cpu().numpy())
        all_outputs.append(y_pred.cpu().numpy())
        all_masks.append( mask.cpu().numpy() )    # ← append here

    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
    return avg_loss, all_inputs, all_outputs, all_masks   # ← return masks

    


# ── Correct extract_latents logic ──
def extract_latents(model, loader, device):
    latents = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_filled = batch['data'].to(device)                     
            # grab the final hidden state per window
            z = model.get_latent_representation(x_filled)         # (batch_size, latent_dim)
            latents.append(z.cpu().numpy())

    # concatenate all batches into one (num_seq, latent_dim) array
    return np.concatenate(latents, axis=0)


