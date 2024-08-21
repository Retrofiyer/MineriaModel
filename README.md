# Dog Breed Classification Model

This repository contains a PyTorch model designed to classify dog breeds based on images. The model is capable of distinguishing between the following breeds:

- **Cocker Spaniel**
- **Pekinese**
- **Poodle**
- **Schnauzer**

## Model Details

- **Architecture**: ResNet-18
- **Number of Classes**: 4
- **Input Image Size**: 224x224
- **Normalization**: The input images are normalized using the following mean and standard deviation values:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`

## Requirements

- Python 3.8 or higher
- PyTorch
- torchvision
- Pillow

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Retrofiyer/MineriaModel.git
    cd MineriaModel
    ```

2. Create a virtual environment and install the dependencies:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Execute FlaskApp:

    ```bash
    python main.py
    ```
