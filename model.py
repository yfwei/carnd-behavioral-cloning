import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D
from keras.callbacks import ModelCheckpoint

lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read and augment the image
def readAndAugmentData(line):
    images = []
    measurements = []

    # steering angle correction value for the left the right camera
    correction = 0.1
    for i in range(3):
        measurement = float(line[3])

        # Exclude 0 steering angle
        if measurement == 0.0:
            break;

        if i == 1:  # Left camera
            measurement += correction
        elif i == 2: # Right camera
            measurement -= correction
        measurements.append(measurement)

        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)

        # The image read by the OpenCV is in BGR color format, but the color
        # format used in the drive.py is in RGB. So we need to convert the color
        # format here to have the same color format when performing steering
        # angle predictions
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

        # Flip the image horizontally
        images.append(cv2.flip(image, 1))
        measurements.append(measurement * -1.0)

    return images, measurements

def showModelOutput(sample, model):
    image = model.predict(sample[None, :, :, :], batch_size=1)[0]

    fig = plt.figure(figsize=(10, 3))
    subplot = fig.add_subplot(121)
    subplot.set_title("Origin")
    subplot.imshow(sample)

    subplot = fig.add_subplot(122)
    subplot.set_title("Preprocessed")
    subplot.imshow(image)
    plt.show()

images = []
measurements = []

# Use multiple processes to read and augment images
with ProcessPoolExecutor() as executor:
    for x, y in executor.map(readAndAugmentData, lines, chunksize=12):
        images.extend(x)
        measurements.extend(y)

X_train = np.array(images)
y_train = np.array(measurements)
assert(len(X_train) == len(y_train))
print("Number of training samples: ", len(X_train))

# Based on the Nvidia End-to-End learning network architecture
model = Sequential()

# Crop the top and bottom portion of the image to remove useless information
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

# Normalize the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))

# The dropout layer to reduce overfitting
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# Output the network summary
model.summary()

# Use Mean Squared Error as the loss function and the Adam optimizer to avoid
# adjust the learning rate maually.
model.compile(loss='mse', optimizer='adam')

# The best parameters wins
check_point = ModelCheckpoint("model.h5", save_best_only=True)

# Train the model
history_object = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=10, batch_size=128, shuffle=True,
                    callbacks=[check_point])

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
