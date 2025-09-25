import torchvision
from torchinfo import summary

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
# .DEFAULT means the best available weights from pretraining on ImageNet

# To access the transforms associated with our weights, we can use the transform() method
# To get transforms automatically
auto_transforms = weights.transforms()
model = torchvision.models.efficientnet_b0(weights=weights)

if __name__ == "__main__":
    print(model)
    summary(model=model,
            input_size=(32,3,244,244),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )
