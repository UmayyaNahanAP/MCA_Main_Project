# # import matplotlib.pyplot as plt
# import tensorflow as tf
# # import pandas as pd
# # import numpy as np

# import warnings
# warnings.filterwarnings('ignore')

# from tensorflow import keras
# from keras import layers
# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import Dropout, Flatten, Dense
# # from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.utils import image_dataset_from_directory
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from tensorflow.keras.preprocessing import image_dataset_from_directory

# import os
# import matplotlib.image as mpimg




import tensorflow as tf
import os
import warnings
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dropout, Flatten, Dense ,Conv2D, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory

warnings.filterwarnings('ignore')


# For local system
path = 'dataset/train'

classes = os.listdir(path)
print(classes)

# Define the directories for the X-ray images
PNEUMONIA_dir = os.path.join(path + '/' + classes[0])
NORMAL_dir = os.path.join(path + '/' + classes[1])

# Create lists of the file names in each directory
pneumonia_names = os.listdir(PNEUMONIA_dir)
normal_names = os.listdir(NORMAL_dir)

print('There are ', len(pneumonia_names),'images of pneumonia infected in training dataset')
print('There are ', len(normal_names), 'normal images in training dataset')


Train = keras.utils.image_dataset_from_directory(
	directory='dataset/train',
	labels="inferred",
	label_mode="categorical",
	batch_size=32,
	image_size=(256, 256))
Test = keras.utils.image_dataset_from_directory(
	directory='dataset/test',
	labels="inferred",
	label_mode="categorical",
	batch_size=32,
	image_size=(256, 256))
Validation = keras.utils.image_dataset_from_directory(
	directory='dataset/val',
	labels="inferred",
	label_mode="categorical",
	batch_size=32,
	image_size=(256, 256))


model = tf.keras.models.Sequential([
	layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
	layers.MaxPooling2D(2, 2),
	layers.Conv2D(64, (3, 3), activation='relu'),
	layers.MaxPooling2D(2, 2),
	layers.Conv2D(64, (3, 3), activation='relu'),
	layers.MaxPooling2D(2, 2),
	layers.Conv2D(64, (3, 3), activation='relu'),
	layers.MaxPooling2D(2, 2),

	layers.Flatten(),
	layers.Dense(512, activation='relu'),
	layers.BatchNormalization(),
	layers.Dense(512, activation='relu'),
	layers.Dropout(0.1),
	layers.BatchNormalization(),
	layers.Dense(512, activation='relu'),
	layers.Dropout(0.2),
	layers.BatchNormalization(),
	layers.Dense(512, activation='relu'),
	layers.Dropout(0.2),
	layers.BatchNormalization(),
	layers.Dense(2, activation='sigmoid')
])


model.compile(
	# specify the loss function to use during training
	loss='binary_crossentropy',
	# specify the optimizer algorithm to use during training
	optimizer='adam',
	# specify the evaluation metrics to use during training
	metrics=['accuracy']
)

history = model.fit(Train,
		epochs=10,
		validation_data=Validation)

model.save('pneumonia_cnn_model.h5')
