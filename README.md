**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/cameras.png "Multiple Cameras"
[image2]: ./images/cropped.png "Cropped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode with maximum speed 30 MPH
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* video.mp4 recording of the vehicle driving autonomously one lap around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidia network for end to end learning. The paper can be found [here](https://arxiv.org/pdf/1604.07316.pdf).

The data is normalized in the model using a Keras Lambda layer (model.py line 79).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 93). 

The 20% of the training data was used to validate the model to ensure the model was not overfitting. (model.py line 111).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving and smooth curve driving.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

#### 1. Solution Design Approach

My very first model was a vanilla Nvidia end to end learning network for self driving cars with only the center camera images. It had a low mean squared error on both training data set and validation data set, but it would drive into the river directly on the first curve.

So to prevent the vehicle from drowning itself, I added left and right camera images to the training data set with 

In order to reduce overfitting. 

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
