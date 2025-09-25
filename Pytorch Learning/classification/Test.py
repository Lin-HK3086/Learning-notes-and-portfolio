import Model, Data
import torch
from torch import nn
import matplotlib.pyplot as plt

model_test = Model.model
model_test.load_state_dict(torch.load('MODEL/01_circle_model.pth'))
model_test.eval()
with torch.inference_mode():
    y_preds = torch.round(model_test(Data.X_test)).squeeze()
print(Model.accuracy_fn(y_preds, Data.y_test))
# Plot decision boundaries for training and test sets

print(y_preds[:10],Data.y_test[:10])
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo
from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Test")
plot_decision_boundary(model_test, Data.X_test, Data.y_test) # model_3 = has non-linearity
plt.show()