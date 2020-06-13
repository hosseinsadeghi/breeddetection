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

## Navigating the app

You'll find a button `upload file` that allows you to upload your file of interest. Then click upload. If successful you'll see a message indicating that the
upload was successful. After a few seconds, you'll see your image as well as the image of a dog that looks like the dog image or face image that you uploaded.

## Credit

Many of the basic codes are taken from this repo from [Udacity](https://github.com/udacity/dog-project.git).

The snapshot of the app contains an image of [Weird Al](https://images.app.goo.gl/YDcdvvrA5thoPNDL8).