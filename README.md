# Brain Tumor Segmentation using U-Net

This project implements a U-Net model for segmenting brain tumors from MRI images using TensorFlow and Keras. The notebook covers the entire workflow from data loading and preprocessing to model training, evaluation, and visualization of the results.

## 📖 Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

This project uses several Python libraries. Make sure you have them installed in your environment.

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn pandas jupyter
```

## Dataset

The model is trained on a brain tumor segmentation dataset, which is expected to be in a zip file.

### Setup Instructions:

1. **Download the dataset**: Ensure you have the `demo_brain_tumor_seg.zip` file.

2. **Set the path**: The notebook expects the dataset to be located in Google Drive at `/content/drive/MyDrive/demo_brain_tumor_seg.zip`. You will need to either:
   - Place the zip file in that exact Google Drive location
   - Modify the path in the first code cell of the notebook to point to the location of your zip file

3. **Dataset Structure**: The dataset should be structured with `train`, `valid`, and `test` directories, each containing `image` and `masks` subdirectories.

```
demo_brain_tumor_seg/
├── train/
│   ├── image/
│   └── masks/
├── valid/
│   ├── image/
│   └── masks/
└── test/
    ├── image/
    └── masks/
```

## 💻 Usage

To run this project, simply open the `Demo_UNET.ipynb` notebook in a Jupyter environment and execute the cells in order.

### Workflow:

1. **Unzip the dataset**: The first cell extracts the images and masks from the zip file
2. **Load and preprocess data**: The subsequent cells load the images, resize them to 128x128, and normalize them
3. **Build and compile the U-Net model**: The U-Net architecture is defined and compiled with the Adam optimizer and binary cross-entropy loss
4. **Train the model**: The model is trained for 15 epochs on the training dataset
5. **Evaluate the model**: The final cells evaluate the model's performance on the test set using Mean Intersection over Union (IoU) and display prediction examples

## 🧠 Model Architecture

The model used is a **U-Net**, which is a type of convolutional neural network (CNN) designed for fast and precise image segmentation. It consists of an encoder (downsampling path) and a decoder (upsampling path) with skip connections.

### Key Components:

- **Encoder**: Captures the context in the image. It is a stack of convolutional and max pooling layers
- **Decoder**: Enables precise localization using transposed convolutions
- **Skip Connections**: Connections between the encoder and decoder paths allow the network to use features from earlier layers, leading to more accurate segmentation

The model summary is available in the notebook, detailing each layer's output shape and number of parameters.

## 📊 Results

The model was trained for 15 epochs and achieved the following performance on the validation and test sets:

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | ~99.48% |
| **Validation Loss** | ~0.0152 |
| **Mean IoU on Test Set** | 0.914 |
| **Average F1 Score on Test Set** | 0.980 |

### Performance Analysis:

- The training history plots for loss and accuracy are included in the notebook, showing that the model converges well without significant overfitting
- An example of the model's prediction on a test image demonstrates the segmentation quality

## 🙌 Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to:

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🔗 References

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- TensorFlow/Keras Documentation
- Brain Tumor Segmentation Research Papers

---

**Note**: Make sure to have sufficient computational resources (preferably GPU) for training the model efficiently.
