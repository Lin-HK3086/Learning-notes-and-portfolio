import torch

import GetDataReady
import ModelFitting

# 1. Set the model in evaluation mode
ModelFitting.model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = ModelFitting.model_0(GetDataReady.X_test)
print(y_preds)
GetDataReady.plot_predictions(predictions=y_preds)