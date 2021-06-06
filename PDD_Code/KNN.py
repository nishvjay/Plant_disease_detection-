import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
from keras.preprocessing.image import img_to_array
from os import listdir
from sklearn.preprocessing import LabelBinarizer
import pickle

# Path to the dataset folder
dir = 'D:/MS/Sem 4/Data 3'

# Resizing images to (150,150) and converting to numpy array
def img_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (150, 150))
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

X_vec = img_L
y_vec = image_labels[:, 0]

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer

class pipestep(object):

    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)

makegray = pipestep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten = pipestep(lambda img_list: [img.ravel() for img in img_list])

k = 21

# Building model
pipeline = Pipeline([
    ('Flatten Image', flatten),
    ('Normalize', Normalizer()),
    ('Classify', KNeighborsClassifier(n_neighbors=k)),
                              ])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                    train_size=0.75)

# Fitting model
pipeline.fit(X_train, y_train)

# Testing model
pipe_pred_test = pipeline.predict(X_test)
pipe_pred_prop = pipeline.predict_proba(X_test)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=pipe_pred_test))

# Accuracy
from sklearn.metrics import accuracy_score
print("Test Accuracy: ", accuracy_score(y_true=y_test, y_pred=pipe_pred_test))

# ......................................................................................................................
# Finding the best 'K' value

accuracy = []
from sklearn import metrics

for k1 in range(1, 25):
    pipeline1 = Pipeline([
        ('Flatten Image', flatten),
        ('Normalize', Normalizer()),
        ('Classify', KNeighborsClassifier(n_neighbors=k1)),
    ])

    neighbors = pipeline1.fit(X_train, y_train)
    pred = neighbors.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, pred))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 25), accuracy, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(accuracy), "at K =", accuracy.index(max(accuracy)))
plt.show()

# Finding minimum error

error_rate = []
for k1 in range(1, 25):
    pipeline1 = Pipeline([
        ('Flatten Image', flatten),
        ('Normalize', Normalizer()),
        ('Classify', KNeighborsClassifier(n_neighbors=k1)),
    ])

    neighbors = pipeline1.fit(X_train, y_train)
    pred = neighbors.predict(X_test)
    error_rate.append(np.mean(pred != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1, 25),error_rate,color='blue', linestyle='dashed',
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
plt.show()

# ......................................................................................................................
# LIME Implementation

from lime import lime_image
explainer = lime_image.LimeImageExplainer()

import skimage
from skimage import transform
from skimage import io
from tensorflow.keras.preprocessing import image

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
predict = pipeline.predict(images)
print("Prediction made by KNN: ", label_binarizer.classes_[predict])

# Getting explanations from LIME
explanation = explainer.explain_instance(images[0].astype('double'), pipeline.predict_proba, top_labels=3, hide_color=0, num_samples=1000)

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

