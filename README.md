# **Sparse Autoencoders using MNIST Digit Dataset**

This repository contains code to implement Sparse Autoencoders (AE) using the MNIST digit dataset and perform k-means clustering on the embeddings to evaluate the performance of the k-means algorithm.

## Requirements
To run the code in this repository, you will need the following:

- tensorflow
- python 3.x
- numpy
- matplotlib
- scipy
- Keras
- Numpy

You can install the required packages using the following command:

**pip install tensorflow keras numpy scikit-learn matplotlib**

## Documentation
The implementation details and usage instructions are provided in the "DL_Assign3_Ques1.ipynb" notebook. The notebook includes comments explaining each step of the code.

## Download Dataset
The MNIST digit dataset can be downloaded from the following link: MNIST Dataset

## Directory Hierarchy
```
│   SAE
│   ├── src
│   │   ├── DL_Assign3_Ques1.ipynb
│   Data
│   ├── mnist
```  
**src**: source codes of the SAE

The directory structure is organized as follows:

The "SAE" directory contains the source code and the main notebook.
The "src" directory inside "SAE" contains the "DL_Assign3_Ques1.ipynb" notebook, which contains the implementation of Sparse Autoencoders and k-means clustering.

The "Dataset" directory contains the downloaded MNIST dataset.
Inside the "Dataset" directory, the "mnist" folder stores the extracted MNIST dataset files.

## Implementation Details
## Sparse Autoencoders (AE)

The implementation of Sparse Autoencoders is done in Python using TensorFlow and Keras libraries. The code is provided in the "DL_Assign3_Ques1.ipynb" notebook.

## The steps include:

Importing the required libraries.

* Downloading and extracting the MNIST dataset.
* Loading and preprocessing the MNIST dataset.
* Defining and training the Sparse Autoencoder model with specified hyperparameters.
* Extracting embeddings from the encoder part of the autoencoder.
* Calculating the Mean Squared Error (MSE) as the reconstruction error.

## K-means Clustering

After training the Sparse Autoencoder, k-means clustering is applied to the embeddings. The code uses scikit-learn's KMeans class to perform the clustering.

## Evaluation

To evaluate the performance of the k-means algorithm, the available labels in the dataset are used. The cluster labels obtained from k-means are mapped to the actual predicted labels using the training data.

The clustering accuracy is calculated for both the training and test sets.
