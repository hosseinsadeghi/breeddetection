## Project Overview

Welcome to Dog Breed Detection app. With this web application you'll be able to 
upload an image of your choosing and determine the dog breed. If the image is that of a human, the app will determin the dog breed that looks alike.
  

![Screenshot](appsnapshot.png)

This app uses convolutional neural networks (CNN) and transfer learning to predict the dog bread
using a neural network with 133 outputs (the total number of breeds that we have data available for).

## Basic usage

1. Clone the repository 
```	
git clone https://github.com/hosseinsadeghi/breeddetection.git
cd breeddetection
```

2. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

3. Install all the requirements by
```
python -m pip install -r requirements.txt
```

4. (Optional) Install TensorFlow with GPU support, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

5. Run the app
```
python run.py
```

## Algorithm

To classify dog images we have used InceptionV3 which is an advanced CNN model with Inception layer that consist of three
parallel 1x1 and a 3x3 max pooling layer. Each of the parallel convolutions except one of the 1x1, are followed by 3x3, 
5x5 and 1x1 convolution layers. At the end of each Inception layer the output of the parallel layers are concatenated
to produce the output layer. The 1x1 convolution can be thought of as a dense layer of `num_filters_in x num_filters_out` for each 
pixel. 

![Inception_layer](https://miro.medium.com/max/1400/1*DKjGRDd_lJeUfVlY50ojOA.png)
![Inception](https://miro.medium.com/max/1656/1*uW81y16b-ptBDV8SIT1beQ.png)

The Inception model has approximately seven million parameters. Our data set of dog images has small number of images.
If we wanted to train the classifier directly on our images we would have run into over-fitting issue.

Most images have many features in common such as edges, color combination, shapes, etc. We can use a previously trained 
model of Inception with Imagenet data set with over 14 millions images to transfer some of the knowledge about images to
out own model. It is a common practice in transfer learning to use a pretrained model with extra layers tha adapts to the 
new data and then fine tune the parameters of the larger pretrained network. In this work, however, we will not fine tune
the pre-trained model.

In order to detect the breed of each dog we use a classifier model that uses labeled data to predict the label of each
image. There are 133 breeds in our data set, for this reason we create a neural network with an output size of 133 neurals
each predicting the weight of a categorical distribution.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_2 ( (None, 2048)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               524544    
_________________________________________________________________
dense_4 (Dense)              (None, 133)               34181     
=================================================================
Total params: 558,725
Trainable params: 558,725
Non-trainable params: 0
_________________________________________________________________
```

After transforming the input images with the pre-trained model, we need to learn how to transform the feature maps to
the 133 weights of each class. We first use a global average pooling to collapse all the feature maps to an array of 
size `num_filters` which is 2048 for the Inception module with the top removed.

We further use a dense layer with 256 neurons to be able to learn a complex function of each of the features to better
predict the labels.

## Data set
Our data set consist of 8351 images with 133 distinct categories. We split the data into data of size
6680, 835, and 836 for training, validation, and test, respectively.

The images are of various size, we use preprocessing to match the size of each image to have the shape
(224, 244, 3) which matched the input size of the Inception model.

The model is trained for 100 epochs with batch size of 20. The model is saved every time the performance 
is improved.

We measure the performance by looking at the classification accuracy of the validation data set. Since the 
distribution of classes is only slightly unbalanced, we can safely use accuracy as a measure of performance. 
After selection based on score on validation set, we achieve 77.5% accuracy on the test set.

We can further improve our model by adding batch normalization the dense layers, dropout, and careful
training procedure, for example, decrease learning rate throughout the training to reduce training noise
produce by small batch size.   
 
  
![akita](static/images/Akita_00236.jpg) ![belgian_sheepdog](static/images/Belgian_sheepdog_01494.jpg)   

## Navigating the app

You'll find a button `upload file` that allows you to upload your file of interest. Then click upload. If successful you'll see a message indicating that the
upload was successful. After a few seconds, you'll see your image as well as the image of a dog that looks like the dog image or face image that you uploaded.

## Credit

Many of the basic codes are taken from this repo from [Udacity](https://github.com/udacity/dog-project.git).

The snapshot of the app contains an image of [Weird Al](https://images.app.goo.gl/YDcdvvrA5thoPNDL8).

To train our model we have use the data set from [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).