#!/usr/bin/env python3
# train_delta_net.py
#   Learn Δλ_max from the dataset produced by vdp_sssc_data.py
#   Usage:  python train_delta_net.py deltas.csv

import sys, pathlib, argparse
import numpy as np
import time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# 1.  CLI & data ----------------------------------------------------------------
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train a DNN to predict Δλ_max")
parser.add_argument("csv", type=pathlib.Path,
                    help="CSV with columns lam,x1,x2,log_rho,log_kappa,delta_max")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--batch",  type=int, default=128)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
X_raw, y_raw = data[:, :-1], data[:, -1:]

# ---------------------------------------------------------------------------
# 2.  Feature scaling (mean 0, std 1) ----------------------------------------
# ---------------------------------------------------------------------------
x_mean, x_std = X_raw.mean(axis=0, keepdims=True), X_raw.std(axis=0, keepdims=True)
y_mean, y_std = y_raw.mean(), y_raw.std()        # not strictly needed, handy for later

X = (X_raw - x_mean) / x_std
y = y_raw                                          # keep Δλ in absolute units

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_t, y_t)

# 80 / 20 split
val_frac = 0.2
val_size = int(len(dataset) * val_frac)
train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                generator=torch.Generator().manual_seed(0))

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=len(val_ds))

# ---------------------------------------------------------------------------
# 3.  Tiny network -----------------------------------------------------------
# ---------------------------------------------------------------------------
class DeltaNet(nn.Module):
    def __init__(self, n_in=5, n_h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h),  nn.Tanh(),
            nn.Linear(n_h, 1),    nn.Softplus()  # Δλ_max ≥ 0
        )
    def forward(self, x): return self.net(x)

device = args.device
model  = DeltaNet().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn= nn.MSELoss()

# ---------------------------------------------------------------------------
# 4.  Training loop with simple early stopping -------------------------------
# ---------------------------------------------------------------------------
best_val = float("inf")
patience = 50
wait     = 0

for epoch in range(1, args.epochs + 1):
    # ---- training ----------------------------------------------------------
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    # ---- validation --------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            val_loss = loss_fn(model(xb), yb).item()
            mae      = torch.mean(torch.abs(model(xb) - yb)).item()
    if epoch % 100 == 0 or val_loss < best_val:
        print(f"epoch {epoch:4d}  val_MSE={val_loss:.3e}  val_MAE={mae:.3e}")

    # ---- early-stopping bookkeeping ---------------------------------------
    if val_loss + 1e-6 < best_val:
        best_val = val_loss
        wait = 0
        torch.save({
            "model_state": model.state_dict(),
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std
        }, "delta_net.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stop.")
            break



# ---------------------------------------------------------------------------
# 5.  Final metrics ----------------------------------------------------------
model.load_state_dict(torch.load("delta_net.pt")["model_state"])
model.eval()
with torch.no_grad():
    xb, yb = next(iter(val_dl))
    xb, yb = xb.to(device), yb.to(device)
    pred   = model(xb)
    mae    = torch.mean(torch.abs(pred - yb)).item()
    rmse   = torch.sqrt(loss_fn(pred, yb)).item()

print(f"\nValidation MAE = {mae:.4e}   RMSE = {rmse:.4e}")
print("Saved best checkpoint to delta_net.pt")