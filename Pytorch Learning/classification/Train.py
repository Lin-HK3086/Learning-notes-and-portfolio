import Data, Model
from Model import loss_fn, optimizer
import torch
from pathlib import Path

model=Model.model
# Training
for epoch in range(10000):
    model.train()
    y_pred = model(Data.X_train).squeeze()
    y_out = torch.round(y_pred)
    loss = loss_fn(y_pred, Data.y_train)
    acc = Model.accuracy_fn(y_out, Data.y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        ty_pred = model(Data.X_test).squeeze()
        ty_out = torch.round(ty_pred)
        test_loss = loss_fn(ty_pred, Data.y_test)
        test_acc = Model.accuracy_fn(ty_out, Data.y_test)
        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.5f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.5f}%")
        if epoch ==9999:
            print(ty_out[:10],Data.y_test[:10])

# Loading
model_path = Path("D:\Python Program\Pytorch Learning\classification\MODEL")
model_path.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_circle_model.pth"
MODEL_SAVE_PATH = model_path / MODEL_NAME

torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
