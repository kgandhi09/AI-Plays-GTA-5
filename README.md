# AI plays GTA V

![ca](https://user-images.githubusercontent.com/36654439/106350827-e1a8e400-62a5-11eb-9315-e9c11886c3cd.png)

This project is part of self learning during the summer of 2021 to get my hands on Deep learning. I used semantic image segmentation technique to and built a Convolutional neural network to predict and identify the position of lanes. Furthermore, developed an algorithm to localize the player and generate a trajectory for the vehicle. Finally developed a drive algorithm to drive to vehicle on the generated trajectory.

## Data Collection 
The data used in this project were collected manually from the live gameplay of GTA V. I played GTA V for an hour and recorded the gameplay into mp4 file and created a program, "dataset_capture.py" to capture images on 5 sec interval from the recorded mp4. Around 720 images were captured to be used for training the CNN model.

## Preprocessing
Image Segmentation with CNN involves feeding segments of an image as input to a convolutional neural network, which labels the pixel. So, in order to create the corresponding labels for the raw images, a masking technique is used where entire image is masked with black pixels except the region of interest, which is lanes in our case. I created a program called "create_mask.py" to create labels from raw images by using cv2.fillPoly() function to create the masks.
- Train_X = Raw_images
- Train_Y = Masked_Images (Labels)

## Building the Model
In order to build the CNN model, I used the open source architecture called U-Net. U-Net is a CNN that was developed for the biomedical image segmentation at the Computer Science Department of the University of Freiburg, Germnany.
![u-net-architecture](https://user-images.githubusercontent.com/36654439/106352547-1753ca00-62b2-11eb-8822-a96e312ebaf1.png)

## Trajectory generation
Once the lane is identified/predicted by the CNN model, I retain the information of what pixels from the entire array composes lane (pixels with red color). I simply then take mean between all the lane pixels in one row to get a central value in that particular row. After doing that for every row, a continous trajectory is generated (shown in green color).

![WhatsApp Image 2021-01-30 at 4 29 51 AM](https://user-images.githubusercontent.com/36654439/106352784-d8267880-62b3-11eb-9155-919e35dda321.jpeg)

## Drive Algorithm
W,A,S,D keys are used in GTA V to drive a vehicle. I used a method of keypress in python to drive the vehicle. After localizing the player and generating the trajectory, algorithm measures the relative position of player from the trajectory, if the player is on left side, D key is pressed to move the player in right direction and vice-versa for left direction.

#### For demonstration of the project, please check out: https://www.youtube.com/watch?v=B34bs3lSeac 
