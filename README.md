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

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My very first model was a vanilla Nvidia end to end learning network for self driving cars with only the center camera images. It had a low mean squared error on both training data set and validation data set, but it would drive directly into the river on the first curve.

In order to prevent the vehicle from drowning itself, I added left and right camera images to the training data set with 0.2 steering angle degree correction. Although the vehicle drove much better with new data, the model had a low mean squared error on the training data set, but a high mean squared error on the validation data set, which implies the model was overfitting. I increased the depth of the model (model.py line 96) and added a dropout layer (model.py line 93) to overcome this overfitting problem.

The vehicle still had problems driving around few curves. So I recorded three curve driving to let the model learn how to drive smoothly around curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   							| 
| Normalization     	|  	|
| Cropping     	| Crop top 65 pixels and bottom 20 pixels 	|
| Convolution 5x5				|	2x2 stride, outputs 36x152x24											|
| ReLU					|			
| Convolution 5x5				|	2x2 stride, outputs 16x77x36											|
| ReLU					|			
| Convolution 5x5				|	2x2 stride, outputs 6x37x48											|
| ReLU					|			
| Convolution 3x3				|	1x1 stride, outputs 4x35x64											|
| ReLU					|			
| Convolution 3x3				|	1x1 stride, outputs 2x33x64											|
| ReLU					|			
| Fully connected		| outputs 100        									|
| Dropout		|         									|
| Fully connected		| outputs 50        									|
| Fully connected		| outputs 20        									|
| Fully connected		| outputs 10        									|
| Fully connected		| outputs 1        									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example images of center lane driving from three cameras:

![alt text][image1]

Then I cropped out top and bottom portion of the image to remove the information that is not helpful to train the model. Here is an example of cropped image:

![alt text][image2]
