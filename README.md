Playing Card Classifier with PyTorch

This project trains a deep learning model using PyTorch to classify playing cards from images. It uses the EfficientNet model via the timm library and follows a standard training loop with custom dataset handling and evaluation.

Project Structure

├── CardClassifier.py        # Main script for training and evaluation
├── README.md                # This file
├── /train                   # Training images organized in subfolders per class
├── /valid                   # Validation images organized similarly
├── /test                    # Test images for final evaluation

How to Run

1. Clone the repo / Download the code

cd /your/project/path

2. Make sure you have Python 3 installed

Run the script with:

python3 CardClassifier.py

💡 If you get a "command not found" error, use python3 instead of python.

Requirements

Install the dependencies before running:

pip install torch torchvision timm matplotlib pandas tqdm pillow

Features

Loads custom image datasets using torchvision.datasets.ImageFolder

Wraps datasets in DataLoader for batch training

Uses EfficientNet (via timm) for feature extraction

Fine-tunes final classifier layer for 53 playing card classes

Trains and validates the model over multiple epochs

Evaluates model using accuracy on validation and test sets

Visualizes loss curves over training

Accuracy Evaluation

The script includes a function to evaluate model accuracy:

def evaluate_accuracy(model, dataloader, device):
    # Returns float between 0 and 1

Visualizations

Loss curves are plotted using matplotlib after training:

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

Notes

Ensure your dataset is structured as:

/train/<class_name>/*.jpg
/valid/<class_name>/*.jpg
/test/<class_name>/*.jpg

The model expects 128x128 image size (resized automatically).

Works on CPU or GPU (CUDA enabled).



Happy Training! 

