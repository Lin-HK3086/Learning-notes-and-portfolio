import torch
import ModelFitting
from ModelBuilding import LinearRegressionModel
from pathlib import Path


# Saving a PyTorch model's state_dict()
# 1. Create models directory
MODEL_PATH = Path("D:\Python Program\Pytorch Learning\workflow\Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=ModelFitting.model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

# Loading a saved PyTorch model's state_dict()
# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(ModelFitting.GetDataReady.X_test) # perform a forward pass on the test data with the loaded model
print(loaded_model_preds)