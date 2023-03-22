import cv2
from keras.models import load_model
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
from keras import models
from keras import layers
import time

debugFlag = False

def train_CNN():

    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Define CNN 
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    # Preprocess data
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    # One-hot encode labels
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Compile and train CNN
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=15, batch_size=128)

    # Evaluate model
    _, test_acc = model.evaluate(test_images, test_labels)
    print("Accuracy: " + str(test_acc))

    # Save model to file
    model.save('model.h5')


# Define function to resize image with padding
def resize_with_padding(image):

    # Get the input shape
    height, width = image.shape

    if height == width:
        # Check if the image is already a square, and return if it is
        return image
    elif height > width:
        # Will automically resize the image according to greater height
        diff = height - width
        left = diff // 2
        right = diff - left
        
        # Add padding to the left and right of the image
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image
    else:
        # Will automically resize the image according to greater width
        diff = width - height
        top = diff // 2
        bottom = diff - top

        # Add padding to the top and bottom of the image
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image


# Define function to recognize digits in an image
def recognize_digits(image):

    # Convert image to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blurring to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to binarize image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of all objects in image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCount = 0
    spacing = 50
    prevW = 0

    for contour in contours:

        # Get bounding box of contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Ensure contour is large enough to be a digit
        if w >= 10 and h >= 10 and w <= 500 and h <= 500:

            digitCount += 1

            # Extract digit from image and preprocess it for recognition
            digit_img = thresh[y:y+h, x:x+w]
            digit_img = resize_with_padding(digit_img)

            name = f'Image before resizing to 28x28 {digitCount}'
            cv2.imshow(name, digit_img)
            cv2.moveWindow(name, prevW + spacing, 0)

            prevW += w + 25

            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = np.reshape(digit_img, (1, 28, 28, 1))
            digit_img = digit_img / 255.0
            digit = np.argmax(model.predict(digit_img, verbose = 0), axis=-1)[0]

            # Draw bounding box around digit and label it with recognized digit
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, str(digit), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Display image with bounding boxes and labels
    cv2.imshow('Digits Recognition', image)
    cv2.waitKey(1)


# Define function to process video feed or file
def process_video(source):
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        recognize_digits(frame)
        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if not debugFlag:
    # Ask user if they want to train the CNN
    train = input("Do you want to train the CNN? (y/n): ")
    if train == 'y' or train == 'Y' or train == 'yes' or train == 'Yes' or train == 'YES':
        train_CNN()

    # Load trained model
    model = load_model('model.h5')

    # # Ask user if they want to process a video file, webcam feed or exit the program
    video = input("\n\nType \n\t1 to process a video file\n\t2 to process a webcam feed\n\tAny other character to exit\n\n>>> ")
    if video == '1':
        video = input("Enter path to video file: ")
    elif video == '2':
        video = 0
    else:
        print("Exiting...")
        exit()
        
    # Process video feed or file
    process_video(video)

else:
    # Debugging/testing purposes
    model = load_model('model.h5')
    process_video("videos/digits.mp4")

