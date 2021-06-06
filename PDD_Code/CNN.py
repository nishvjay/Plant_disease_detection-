import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# Path to the dataset folder
dir = 'D:/MS/Sem 4/Data 3'

# Resizing images to (150,150) and converting to numpy array
def img_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (150,150))
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

# Getting 1000 images of tomato leaves from each folder from the plant village dataset
img_L, lab_L = [], []

try:
    plant_disease_folder_list = listdir(dir)
    for plant_disease_folder in plant_disease_folder_list:
        plant_disease_image_list = listdir(f"{dir}/{plant_disease_folder}/")
        for image in plant_disease_image_list[:1000]:
            img_dir = f"{dir}/{plant_disease_folder}/{image}"
            if img_dir.endswith(".jpg")==True or img_dir.endswith(".JPG")==True:
                img_L.append(img_array(img_dir))
                lab_L.append(plant_disease_folder)
except Exception as e:
    print(f"Error : {e}")

# Check the number of images loaded
image_length = len(img_L)
print(f"Total number of images loaded: {image_length}")

# Converting labels into binary form
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(lab_L)
total_no_classes = len(label_binarizer.classes_)

print("Total number of classes loaded: ", total_no_classes)

# Transform the loaded image data into numpy array
np_image_list = np.array(img_L, dtype=np.float16) / 225.0

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state=42)

# Parameters
EPOCHS = 25
LR = 1e-3
BATCH_SIZE = 256
W = 150
H = 150
D = 3

# Building model
model = Sequential()
inputShape = (H, W, D)
chanDim = -1

if K.image_data_format() == "channels_first":
    inputShape = (D, H, W)
    chanDim = 1

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(total_no_classes))
model.add(Activation("softmax"))

model.summary()

# Initialize optimizer
opt = Adam(lr=LR, decay=LR / EPOCHS)

# Compile model
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                              validation_data=(x_test, y_test),
                              epochs=EPOCHS,
                              verbose=1)

accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

# Train and validation accuracy
plt.plot(epochs, accuracy, 'b', label='Training accurarcy')
plt.plot(epochs, validation_accuracy, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()

# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, validation_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# Accuracy
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

rounded_predictions = model.predict_classes(x_test, batch_size=32, verbose=0)

rounded_labels=np.argmax(y_test, axis=1)

# Classification report
from sklearn.metrics import classification_report
cr = classification_report(rounded_labels, rounded_predictions)
print(cr)

# ......................................................................................................................
# LIME Implementation

import skimage.io
import skimage.segmentation
import numpy as np
from lime import lime_image
import skimage
from skimage import transform
from skimage import io
from tensorflow.keras.preprocessing import image

explainer = lime_image.LimeImageExplainer()

# Image you want to predict and return explanations for
url = 'C:/Users/nisha/Desktop/New folder/test_area/image (1002).JPG'

def transform_img(url):
    img = skimage.io.imread(url)
    img = skimage.transform.resize(img, (150, 150))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img

images = transform_img(url)

# Checking what the model predicts
predict = model.predict_classes(images)
# print(predict)
print("Prediction made by CNN: ", label_binarizer.classes_[predict][0])

# Getting explanations from LIME
explanation = explainer.explain_instance(images[0].astype('double'), model.predict_proba, top_labels=3, hide_color=0, num_samples=1000)

from skimage.segmentation import mark_boundaries

# Visualizing explanations
temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
ax1.imshow(mark_boundaries(temp_1, mask_1))
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax1.axis('off')
ax2.axis('off')
plt.show()