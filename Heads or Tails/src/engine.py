import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import os

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y =X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred_class)
    train_acc = train_acc / len(dataloader)
    train_loss = train_loss / len(dataloader)
    return train_loss, train_acc

def valid_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_valid = model(X)
            valid_loss += loss_fn(y_valid, y).item()
            y_valid_class = torch.argmax(y_valid, dim=1)
            valid_acc += (y_valid_class == y).sum().item()/len(y_valid_class)
    valid_acc = valid_acc / len(dataloader)
    valid_loss = valid_loss / len(dataloader)
    return valid_loss, valid_acc

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    results = {"train_acc": [], "valid_acc": [],
               "train_loss": [], "valid_loss": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{epochs}"
              f"\tTrain Acc:\t{train_acc:.4f}"
              f"\tTrain Loss: {train_loss:.3f}"
              f"\tValid Acc: {valid_acc:.4f}"
              f"\tValid Loss: {valid_loss:.3f}")
        results["train_acc"].append(train_acc)
        results["valid_acc"].append(valid_acc)
        results["train_loss"].append(train_loss)
        results["valid_loss"].append(valid_loss)
    return results


