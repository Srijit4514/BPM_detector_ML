import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.train_utils import RPPGDataset
from src.loss import neg_pearson_loss
from src.model import PhysNetED  # from official repo file

# --- Hyperparams ---
LR = 1e-4
EPOCHS = 60
BATCH = 4

# --- Load train + validation ---
train_data = RPPGDataset("data/UBFC-rPPG/train_vid", "data/UBFC-rPPG/train_ppg")
val_data   = RPPGDataset("data/UBFC-rPPG/val_vid", "data/UBFC-rPPG/val_ppg")

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PhysNetED().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for vids, ppgs in train_loader:
        vids = vids.to(device)
        ppgs = ppgs.to(device)

        optimizer.zero_grad()
        pred = model(vids).squeeze(1)
        loss = neg_pearson_loss(pred, ppgs)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader)}")

    # --- (Optional) validation loop here...
