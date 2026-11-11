"""Gradient training shared by every model: Adam, LR scheduler, early stopping."""
import copy

import torch
from torch import nn

from . import config as C


def train_adam(model, X_tr, y_tr, X_va, y_va, lr=C.ADAM_LR, max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE):
    """Full-batch training. Returns the loss history; the model ends at its best validation epoch."""
    X_tr, y_tr, X_va, y_va = (torch.as_tensor(a, dtype=torch.float32) for a in (X_tr, y_tr, X_va, y_va))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=C.LR_PATIENCE)
    loss_fn = nn.MSELoss()

    def val_loss():
        model.eval()
        with torch.no_grad():
            return loss_fn(model(X_va), y_va).item()

    # one training-mode pass so that 'batch' attention summaries exist before the first evaluation
    model.train()
    with torch.no_grad():
        train0 = loss_fn(model(X_tr), y_tr).item()
    best, best_state, wait = val_loss(), copy.deepcopy(model.state_dict()), 0
    history = {"train": [train0], "val": [best]}
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        v = val_loss()
        sched.step(v)
        history["train"].append(loss.item())
        history["val"].append(v)
        if v < best - 1e-7:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return history


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy()
