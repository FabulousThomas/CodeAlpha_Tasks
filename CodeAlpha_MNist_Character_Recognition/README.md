## Handwritten Character Recognition

### Objective
To develop a Deep Learning model capable of recognizing and classifying handwritten digits (0-9) using the MNIST dataset.

### Implementation Details
* **Framework:** **PyTorch** (Object-Oriented Architecture).
* **Model Architecture:** Convolutional Neural Network (CNN).
    * **Layer 1:** 2D Convolution (32 filters) + ReLU + Max-Pooling.
    * **Layer 2:** 2D Convolution (64 filters) + ReLU + Max-Pooling.
    * **Fully Connected:** 128 neurons followed by a 10-class Softmax output.
* **Optimization:** Adam Optimizer and Cross-Entropy Loss.
* **Data Handling:** Implemented `torchvision` transforms for normalization and data batching via `DataLoader`.

### Results
* **Test Accuracy:** ~98.5% - 99.0%.
* **Insight:** The CNN architecture successfully captures spatial hierarchies in image data, significantly outperforming traditional feed-forward networks.