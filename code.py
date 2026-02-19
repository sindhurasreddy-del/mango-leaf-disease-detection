#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install iamge')
get_ipython().system('pip install keras')
get_ipython().system('pip install matplotlib')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

main_directory =  r"C:\Users\ACER\Downloads\mango"
images = []
imagesByCategory = {}

import os
for dirname, _, filenames in os.walk( r"C:\Users\ACER\Downloads\mango"):
    dirname
    for filename in filenames:
        images.append(os.path.join(dirname, filename))
        
        ls = dirname.split('/')
        if(ls[-1] not in imagesByCategory):
            imagesByCategory[ls[-1]] = [os.path.join(dirname, filename)]
        else:
            imagesByCategory[ls[-1]].append(os.path.join(dirname, filename))


# In[2]:


print("Total images: " + str(len(images)))
print('')
print("Total categories: " + str(len(imagesByCategory.keys())))
print('')
print("Categories: " + str(', '.join(imagesByCategory.keys())))
print('')

categories = list(imagesByCategory.keys())
categories.sort()
print("Sorted Categories: " + str(', '.join(categories)))


# In[3]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint


# In[4]:


BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNELS=3
EPOCHS=3


# In[5]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
     r"C:\Users\ACER\Downloads\mango",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[6]:


class_names = dataset.class_names
class_names


# In[7]:


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# # Visualize some of the images

# In[8]:


plt.figure(figsize=(15, 20))
for image_batch, labels_batch in dataset.take(3):
    for i in range(12):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
#         plt.axis("off")


# # Function to Split Dataset
## Dataset should be bifurcated into 3 subsets, namely:

## Training: Dataset to be used while training
## Validation: Dataset to be tested against while training
## Test: Dataset to be tested against after we trained a model
# In[9]:


len(dataset)


# In[10]:


train_size = 0.8
len(dataset)*train_size


# In[11]:


train_ds = dataset.take(100)
len(train_ds)


# In[12]:


test_ds = dataset.skip(100)
len(test_ds)


# In[13]:


val_size=0.1
len(dataset)*val_size


# In[14]:


val_ds = test_ds.take(12)
len(val_ds)


# In[15]:


test_ds = test_ds.skip(12)
len(test_ds)


# In[16]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


# In[17]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[18]:


len(train_ds)


# # Cache, Shuffle, and Prefetch the Dataset

# In[19]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# # Building the Model

# Creating a Layer for Resizing and Normalization

# In[20]:


resize_and_rescale =  tf.keras.Sequential([
  layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.Rescaling(1./255),
])


# In[21]:


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])


# In[22]:


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[23]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)


# In[24]:


model.summary()


# In[25]:





model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[26]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
   
)
model.save('saved/model.h5')


# In[27]:


scores = model.evaluate(test_ds)


# In[28]:


scores


# In[29]:


history


# In[30]:


history.params


# In[31]:


history.history.keys()


# In[32]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[33]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[34]:


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[35]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define constants
IMAGE_SIZE = 224
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Define cure steps for each disease
cure_steps = {
    'Anthracnose': [
        "Remove affected parts: Prune and destroy affected leaves and plant parts.",
        "Apply fungicides: Use fungicides containing chlorothalonil, copper, or mancozeb.",
        "Maintain sanitation: Clean the area around plants of plant debris.",
        "Water properly: Avoid overhead watering."
    ],
    'Bacterial Canker': [
        "Remove infected plants: Uproot and destroy infected plants.",
        "Use copper sprays: Apply copper-based bactericides.",
        "Sanitize tools: Disinfect pruning tools regularly.",
        "Avoid overhead irrigation: Water at the base of plants."
    ],
    'Cutting Weevil': [
        "Manual removal: Handpick weevils and larvae.",
        "Apply insecticides: Use insecticides like pyrethroids or neem oil.",
        "Use beneficial nematodes: Introduce beneficial nematodes in the soil."
    ],
    'Die Back': [
        "Pruning: Prune and destroy affected plant parts.",
        "Apply fungicides: Use appropriate fungicides.",
        "Improve air circulation: Ensure proper spacing of plants.",
        "Water management: Avoid overwatering and ensure good drainage."
    ],
    'Gall Midge': [
        "Remove galls: Prune and destroy affected plant parts.",
        "Apply insecticides: Use insecticides that target gall midges.",
        "Use sticky traps: Monitor and reduce midge populations with sticky traps."
    ],
    'Healthy': [
        "Regular monitoring: Inspect plants frequently.",
        "Balanced nutrition: Provide balanced fertilization.",
        "Water management: Water plants appropriately.",
        "Mulching: Use mulch to conserve moisture and reduce weed growth."
    ],
    'Powdery Mildew': [
        "Remove infected parts: Trim and dispose of infected leaves.",
        "Apply fungicides: Use fungicides like sulfur, neem oil, or potassium bicarbonate.",
        "Increase airflow: Space plants properly and prune to improve air circulation.",
        "Water at ground level: Avoid wetting foliage when watering."
    ],
    'Sooty Mould': [
        "Control insects: Manage sap-sucking insects like aphids and whiteflies.",
        "Clean leaves: Wash affected leaves with water.",
        "Use insecticidal soap: Apply insecticidal soaps or oils.",
        "Prune affected areas: Trim and remove heavily infested plant parts."
    ]
}

# Function to predict the class of an image
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)
# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path2):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path2, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path3):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path3, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path4):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path4, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path5):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path5, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path6):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path6, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path7):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path7, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path8):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path8, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Example of using the load_and_predict function
image_path = r"C:\Users\ACER\Downloads\mango\Anthracnose\IMG_20211011_164417 (Custom).jpg"

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path)

# Example of using the load_and_predict function
image_path2 = r"C:\Users\ACER\Downloads\mango\Bacterial Canker\IMG_20211106_143244 (Custom).jpg"

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path2)

# Example of using the load_and_predict function
image_path3 = r"C:\Users\ACER\Downloads\mango\Cutting Weevil\20211011_162501 (Custom).jpg"

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path3)

# Example of using the load_and_predict function
image_path4 = r"C:\Users\ACER\Downloads\mango\Die Back\IMG_20211027_193728 (Custom).jpg" 

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path4)

# Example of using the load_and_predict function
image_path5 = r"C:\Users\ACER\Downloads\mango\Gall Midge\IMG_20211106_170036 (Custom).jpg"

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path5)

# Example of using the load_and_predict function
image_path6 = r"C:\Users\ACER\Downloads\mango\Healthy\20211231_162248 (Custom).jpg"

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path6)

# Example of using the load_and_predict function
image_path7 = r"C:\Users\ACER\Downloads\mango\Powdery Mildew\IMG_20211107_130036 (Custom).jpg"

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path7)

# Example of using the load_and_predict function
image_path8 = r"C:\Users\ACER\Downloads\mango\Sooty Mould\IMG_20211212_150321 (Custom).jpg"

load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path8)


# In[43]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define constants
IMAGE_SIZE = 224
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Define cure steps for each disease
cure_steps = {
    'Anthracnose': [
        "Remove affected parts: Prune and destroy affected leaves and plant parts.",
        "Apply fungicides: Use fungicides containing chlorothalonil, copper, or mancozeb.",
        "Maintain sanitation: Clean the area around plants of plant debris.",
        "Water properly: Avoid overhead watering."
    ],
    'Bacterial Canker': [
        "Remove infected plants: Uproot and destroy infected plants.",
        "Use copper sprays: Apply copper-based bactericides.",
        "Sanitize tools: Disinfect pruning tools regularly.",
        "Avoid overhead irrigation: Water at the base of plants."
    ],
    'Cutting Weevil': [
        "Manual removal: Handpick weevils and larvae.",
        "Apply insecticides: Use insecticides like pyrethroids or neem oil.",
        "Use beneficial nematodes: Introduce beneficial nematodes in the soil."
    ],
    'Die Back': [
        "Pruning: Prune and destroy affected plant parts.",
        "Apply fungicides: Use appropriate fungicides.",
        "Improve air circulation: Ensure proper spacing of plants.",
        "Water management: Avoid overwatering and ensure good drainage."
    ],
    'Gall Midge': [
        "Remove galls: Prune and destroy affected plant parts.",
        "Apply insecticides: Use insecticides that target gall midges.",
        "Use sticky traps: Monitor and reduce midge populations with sticky traps."
    ],
    'Healthy': [
        "Regular monitoring: Inspect plants frequently.",
        "Balanced nutrition: Provide balanced fertilization.",
        "Water management: Water plants appropriately.",
        "Mulching: Use mulch to conserve moisture and reduce weed growth."
    ],
    'Powdery Mildew': [
        "Remove infected parts: Trim and dispose of infected leaves.",
        "Apply fungicides: Use fungicides like sulfur, neem oil, or potassium bicarbonate.",
        "Increase airflow: Space plants properly and prune to improve air circulation.",
        "Water at ground level: Avoid wetting foliage when watering."
    ],
    'Sooty Mould': [
        "Control insects: Manage sap-sucking insects like aphids and whiteflies.",
        "Clean leaves: Wash affected leaves with water.",
        "Use insecticidal soap: Apply insecticidal soaps or oils.",
        "Prune affected areas: Trim and remove heavily infested plant parts."
    ]
}

# Function to predict the class of an image
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)
# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path2):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path2, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path3):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path3, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path4):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path4, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path5):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path5, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path6):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path6, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path7):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path7, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Function to load the model and make predictions for a user-provided image
def load_and_predict_with_cure_steps(model_path, image_path8):
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path8, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure_steps_for_predicted_class = cure_steps.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print("Curative steps for predicted class:")
    for step in cure_steps_for_predicted_class:
        print(step)

# Example of using the load_and_predict function
image_path = r"C:\Users\ACER\Downloads\mango\Anthracnose\IMG_20211011_164651 (Custom).jpg"
load_and_predict_with_cure_steps(r"C:\Users\ACER\Downloads\model.h5", image_path)


# In[37]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.image.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[38]:


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
import pandas as pd
import seaborn as sns


# In[39]:


num_images = 32
# List of class names
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
truth = []
prediction = []

for idx, (images, labels) in enumerate(test_ds.take(13)):
    for i in range(num_images):

        predicted_class,confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        truth.append(actual_class)
        prediction.append(predicted_class)

print("Truth:", truth)
print("Prediction:", prediction)
# Source code credit for this function: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823


# In[40]:


# Source code credit for this function: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix')
    plt.show()


# In[41]:


cm = confusion_matrix(truth,prediction)
print_confusion_matrix(cm,class_names)


# In[42]:


print(classification_report(truth, prediction))







