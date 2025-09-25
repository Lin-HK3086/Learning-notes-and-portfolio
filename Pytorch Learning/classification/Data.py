from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Make 1000 samples
n_samples = 1000

# Create circles(contain two kinds of datas)
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values

def DataVisualization():
    print(f"First 5 X features:\n{X[:5]}")
    print(f"\nFirst 5 y labels:\n{y[:5]}")

    # Make DataFrame of circle data
    circles = pd.DataFrame({"X1": X[:, 0],
                            "X2": X[:, 1],
                            "label": y
                            })
    print(circles.head(10))

    # Check different labels
    print(circles.label.value_counts())

    # Visualize with a plot
    plt.scatter(x=X[:, 0],
                y=X[:, 1],
                c=y,
                cmap=plt.cm.RdYlBu)
    plt.show()
# if you want to show this datas you can run this method
#DataVisualization()

# Turn data into tensors
# Otherwise this causes issues with computations later on
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

device = "cuda"
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
# the datas are ready