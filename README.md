# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The validation loss was copmuted to see if the model was over-fitting the data. More data, especially evasive/recovery maneuvers were collected to ensure the validation loss also dropped along with the training loss. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and clockwise direction driving data. These steps ensured that the model was able to capture the general driving behaviour for Track 1.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I decided to use a simple convolution neural network model as described in the lectures. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I used a combination of center lane driving, recovering from the left and right sides of the road and clockwise direction driving data. These steps gave me more training data to work with and the validation loss also began to drop.

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The input image dimensions are 160x320 in rgb. They are then converted to grascale and the bagkground information is cropped.

Three connvolutional layers are then used: 
First convolutional layer has 24 filters with a 5x5 kernel size. 
Second convolutional layer has 36 filters with a 5x5 kernel size.
Third convolutional layer has 48 filters with a 5x5 kernal size.

The convolutional layers are followed by two Relu activation layers.

Finally, there are 3 fully connected layers of 100, 50, 10 neurons which finally output a single steering value. 


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I used a combination of center lane driving, recovering from the left and right sides of the road and clockwise direction driving data. 

After the collection process, I had around 20,000 data points. I then preprocessed this data by converting it to grayscale, as the rgb image did not offer any advantage or driving related features to be captured. I also cropped the image to remove any unneccessary background information. I finally randomly shuffled the data set and put 20% of the data into a validation set. To augment the data set, I flipped images and angles thinking that this would generalize the driving behaviour in both directions around the track.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation and training losses. I used an adam optimizer so that manually training the learning rate wasn't necessary.
