# FLOWER IMAGE CLASSIFIER
By: S. D. Boadi

## Table of Contents
1. Project Overview
2. File Descriptions
3. Prerequisite
4. How To Run The Application
5. Expected Results
---
## Project Overview
A project work submitted to [Udacity](https://www.udacity.com/ "Udacity Home") in partial fulfillment of the requirements for the nanodegree in AI Programming with Python.

A code was first developed for an image classifier built with PyTorch and then converted it into a command line application. The application uses one of the convolutional neural networks `vgg16` to train the classifier which is capable to identify 102 different flower species. 

---
## File Descriptions
* _Image Classifier Project.ipynb_ contains the code.
* _flowers_ folder contains the dataset to train the network.
* _train.py_ trains a network on a dataset and save the model as a checkpoint.
* _predict.py_ predicts the image respectively.

---
## Prerequisite
You should have Python 3.7 or above installed.
One must use a GPU to achieve "training" in the shortest possible time. Make sure it has cuda support.

---

## How To Run The Application
1. Train a new network<br>
Run the train.py file
```
python train.py --data_dir 'ImageClassifier/flowers' --gpu True
```
Where `'ImageClassifier/flowers'` is the path to the dataset.<br>
The other arguments the user could add are:
  * _Learning rate_: `--lr`, which has a default value of 0.01.
  * _Hidden units for fc layer_: `--hidden_units`, which has a default value of 512.
  * _Number of epochs_: `--epochs`, which has a defualt value of 20.
  
The code loads a pre-trained network to train a new, untrained neural network as a classifier. The trained network is tested on the dataset to give a good accuracy estimate for the model's perfomance. The trained model would be saved as a checkpoint, _my_checkpoint.pth_.

---
2. Predict flower name<br>
Run the predict.py file
```
python predict.py --image_pth 'Images/flowerimage_06621' --gpu True
```
Where `'Images/flowerimage_06621'` is the path to an image of a flower.<br>
The other arguments the user could add are:
  * _Path of saved model_: `--checkpoint` 
  * _Display top k probabilities_: `--topk`, which has a default value of 3.

The code loads a saved model to predict the name of flower from an image. The image is first preprocessed before it can be used as input for the model.

---
## Expected Results
The application tells the name of flower from an image.