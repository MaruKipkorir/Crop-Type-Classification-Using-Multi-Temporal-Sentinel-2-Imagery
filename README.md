# Crop-Type-Classification-Using-Multi-Temporal-Sentinel-2-Imagery

## Project Overview

This project leverages remote sensing and deep learning to automate crop classification at the **pixel level**, enabling large-scale and precise agricultural monitoring.

The goal is to build a classification model that can distinguish between **cocoa**, **rubber**, and **oil palm** plantations using **multi-temporal satellite imagery**. The model is trained on open-source **Sentinel-2 data** from the Copernicus program .

> This project was developed as part of the [Zindi Crop Identification Challenge](https://zindi.africa/competitions/cote-divoire-byte-sized-agriculture-challenge).
Here’s a clean and well-organized **Dataset Description** section for your README, incorporating the structure, multi-temporal nature, and file contents:

---

## Dataset Description
[Dataset](https://zindi.africa/competitions/cote-divoire-byte-sized-agriculture-challenge/data)
The dataset provided for this challenge consists of **multi-temporal Sentinel-2 satellite imagery**, representing different months of the year. Each location (represented by a unique `ID`) includes up to **12 images** corresponding to monthly observations, capturing seasonal crop characteristics.
The goal is to utilize this **temporal dimension** to learn phenological patterns associated with each crop type, as they evolve throughout the year.

### Dataset Structure
The images are organized by ID and time, with each `.tif` file representing a satellite image containing **12 spectral bands**.

* **Training data**: Contains labeled samples (`TrainDataset.csv`) along with corresponding multi-band `.tif` images.
* **Test data**: Contains unlabeled samples (`TestDataset.csv`) with similar image structures, used for generating predictions.
* **Geometries**: Vector data representing the spatial extent of labeled training regions (`trainGeo.zip`).

### Files
`S2Images.zip` Contains all satellite images (train & test) as multi-band `.tif` files.  
`TrainDataset.csv` Metadata and labels for training samples. Includes image paths and crop classes.  
`TestDataset.csv` Metadata for test samples (no labels). Used for generating competition submissions.  
`trainGeo.zip` GeoJSON or shapefile defining training polygon boundaries (optional).  
`ByteSizedAgriStarterBook.ipynb` Starter notebook demonstrating image preprocessing and submission steps.

---

## Approach

### Data Preprocessing

* The training data consists of multi-temporal Sentinel-2 images, each associated with a crop class.
* Each image sample is a `.tif` file containing 12 spectral bands. These bands were rescaled by dividing pixel values by 15,000 to normalize the inputs to range(0-1).
* Metadata from `TrainDataset.csv` was used to link crop labels with corresponding `.tif` images.
* The dataset was split into training and validation sets using stratified sampling to maintain class balance.

### Model Architecture

* A CNN based on the **VGG16** architecture was implemented from scratch and modified to handle 12-channel Sentinel-2 imagery.
* The model includes:

  * 5 convolutional blocks, each consisting of 2–3 `Conv2D` layers followed by `BatchNormalization`, `ReLU` activation, and `MaxPooling`.
  * Two dense layers of size 4096 with batch normalization and `ReLU`, followed by a softmax output layer for classification.
* Input shape was set to `(224, 224, 12)` to accommodate resized 12-band satellite images.

### Training Strategy

* Data was loaded using a custom `SatelliteImageGenerator` that reads `.tif` images in batches, normalizes them, and converts labels to one-hot encoded vectors.
* The model was compiled with:

  * Optimizer: `SGD` with an initial learning rate of `0.001`.
  * Loss Function: `categorical_crossentropy`.
  * Metric: `accuracy`.
* A `ReduceLROnPlateau` callback was used to reduce the learning rate when validation loss plateaus.
* A `ModelCheckpoint` callback saved the best model based on validation loss.

---

## Results

The final model achieved a Weighted Average F1 Score of **88.6%** and was ranked **20th** on the private leaderboard of the challenge. This model performed reasonably well but there remains significant a gap between this score and the top-performing models.

---

## Challenges

This project presented several challenges, particularly due to the complexity of the dataset and the unique nature of multi-spectral, multi-temporal satellite imagery:

* **Limited Understanding of Image Structure**:
  The dataset consisted of 12-band Sentinel-2 images—significantly different from the standard 3-channel (RGB) images commonly used in computer vision. Understanding the meaning and importance of each spectral band, and how to effectively preprocess and utilize them, was a challenge.

* **Lack of Pretrained Models**:
Most popular pretrained convolutional neural networks are designed for 3-channel RGB images. There are currently no widely available pretrained models for 12-channel satellite imagery, which meant I had to train my model from scratch. This limited the performance benefits that typically come from transfer learning.

---

## Notable Observations
### 1. Model Architectures

I experimented with several model architectures during the development process. I began with a simple CNN with three convolutional layers followed by a single dense layer. I then tried a slightly deeper architecture with four convolutional layers and two dense layers. These simpler models trained significantly faster and achieved high scores, with the simplest architecture reaching a score of **86.8%**.

However, since even marginal improvements are critical in a competitive setting, I ultimately chose to proceed with a more complex model (based on the VGG16 structure) that offered slightly better performance despite its longer training time.

### 2. Augmentation

I applied offline augmentation, where each original image was paired with an augmented version, doubling the dataset size. Both the original and augmented images were used during training. However, this approach led to slower training times and worsened model performance, suggesting that the augmentations may have introduced noise or unhelpful variability.

Due to the **12-band** nature of the satellite imagery, most built-in augmentations from `keras` and `keras_cv` were incompatible. I avoided geometric transformations as they were unlikely to provide meaningful variation for this task. The augmentations I used were:

* `RandomSharpness`
* `RandomGaussianBlur`
* `RandomBrightness`
* `RandomContrast`
* `RandomZoom`

### 3. Batch Normalization Behavior

Models trained without Batch Normalization consistently collapsed to predicting a single class. This issue persisted across different architectures and training configurations.

In fact, when I attempted to use only the RGB bands (to match the input expectations of standard pretrained models) and fine-tuned the pretrained VGG16 model from Keras, the model still predicted only one class. This suggested that the problem was not limited to custom architectures. The most likely cause might be unstable gradients which Batch Normalization helps mitigate.

### 4. Optimizer and Learning Rate Scheduling

During training, I experimented with different optimization strategies. I found that using Stochastic Gradient Descent (SGD) combined with a learning rate scheduler (`ReduceLROnPlateau`) consistently outperformed other approaches.

When using SGD with a learning rate scheduler, the model continued to improve even after 50+ epochs, showing stable and gradual learning. When using a constant learning rate or switching to the Adam optimizer, the model typically stopped improving after just a few epochs.

---

