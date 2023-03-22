----------------------------------------------------------------------------------------------
Readme file
Digits Recognition Project using Convolutional Neural Network from keras MNIST dataset
CMPE-258
Prof. Harry Li
Author: Anshul Shandilya (9894)
----------------------------------------------------------------------------------------------

This project is about recognizing digits in an image using OpenCV and a trained Convolutional 
Neural Network (CNN) model. The program reads the video feed from the default camera or a 
video file and recognizes digits in each frame. The recognized digits are marked with a 
bounding box and the recognized digit is labeled.

Prerequisites --------------------------------------------------------------------------------

- Python 3.x
- OpenCV
- Keras with TensorFlow backend (2.x and above)
- Numpy

Usage ----------------------------------------------------------------------------------------

    1. Run "mnist_recognize_digits.py" file by using the command "python recognize_digits.py"

    2. The program will initially ask if you want to train the CNN using MNIST data from 
       karas.dataset.

       Type "y" if you want to train the CNN model
       Type "n" if you have, and want to use a pre-trained model 

       ***Note*** : If you want to use your pre-trianed model, make sure it is in the same 
                    directory as the .py file, and the name of the model file is "model.h5"

    3. The program will then prompt the user for a choice of video camera or a saved file.

       a. Type 1 for saved file option

            If you choose to perform prediction on a saved file, then the program will ask 
            the user to input the path of the video file.

       b. Type 2 for webcam option

            If you choose the webcam option, then use the webcam to show the program the 
            handwritten digits on paper.

       c. Type any other character for exiting the program at this stage.

    4. The program will recognize the digits and label them with a bounding box, along with
       seperate windows for pre-processed steps for individual digits.

    5. To exit the program during recognition stage, press "CTRL + c" at the terminal.

----------------------------------------------------------------------------------------------