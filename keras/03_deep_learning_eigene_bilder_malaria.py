import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

from google.colab import drive
drive.mount('/content/drive')

my_data_dir = '/content/drive/My Drive/dl-udemy/cell_images'

# Check if the directory exists
if not os.path.exists(my_data_dir):
    print(f"Error: Directory '{my_data_dir}' not found.")
else:
    print(f"Successfully mounted and found data directory: {my_data_dir}")

    # Example: List the contents of the directory
    print("Contents of the data directory:")
    for item in os.listdir(my_data_dir):
        print(item)

    # Example: Read an image (replace with your image file)
    image_path = os.path.join( '/content/drive/My Drive/dl-udemy/cell_images/test/parasitized', 'C189P150ThinF_IMG_20151203_141004_cell_74.png')
    if os.path.exists(image_path):
      img = imread(image_path)
      plt.imshow(img)
      plt.show()
    else:
      print(f"Error: Image '{image_path}' not found.")

# my_data_dir = os.path.join('/content', 'cell_images')

# STELLE SICHER, DASS DIE UNTERVERZEICHNISSE 'test' UND 'train' KORREKT GEFUNDEN WERDEN:
os.listdir(my_data_dir)

test_path = os.path.join(my_data_dir, 'test')
train_path = os.path.join(my_data_dir, 'train')

os.listdir(test_path)

os.listdir(train_path)

os.listdir(os.path.join(train_path, 'parasitized'))[0]

para_cell = os.path.join(train_path, 'parasitized', 'C100P61ThinF_IMG_20150918_144104_cell_162.png')

para_img= imread(para_cell)

plt.imshow(para_img)

para_img.shape

unifected_cell_path = os.path.join(train_path, 'uninfected', os.listdir(os.path.join(train_path, 'uninfected'))[0])
unifected_cell = imread(unifected_cell_path)
plt.imshow(unifected_cell)

"""**Lasst uns überprüfen, wie viele Bilder es gibt.**"""

len(os.listdir(os.path.join(train_path, 'parasitized')))

len(os.listdir(os.path.join(train_path, 'uninfected')))

"""**Lasst uns die durchschnittliche Dimension der Bilder bestimmen.**"""

para_img.shape

# Other options: https://stackoverflow.com/questions/1507084/how-to-check-dimensions-of-all-images-in-a-directory-using-python
dim1 = []
dim2 = []
for image_filename in os.listdir(os.path.join(test_path, 'uninfected')):

    img = imread(os.path.join(test_path, 'uninfected', image_filename))
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

# sns.jointplot(dim1,dim2)

np.mean(dim1)

np.mean(dim2)

image_shape = (130,130,3)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

plt.imshow(para_img)

plt.imshow(image_gen.random_transform(para_img))

plt.imshow(image_gen.random_transform(para_img))

image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)

"""# Modell erzeugen"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

#https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary so we use sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

"""## Früh terminieren"""

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=2)

"""## Modell trainieren"""

# help(image_gen.flow_from_directory)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)

train_image_gen.class_indices

import warnings
warnings.filterwarnings('ignore')

results = model.fit(train_image_gen,epochs=20,
                              validation_data=test_image_gen,
                             callbacks=[early_stop])

from tensorflow.keras.models import load_model
model.save('malaria_detector.keras')

"""# Modell evaluieren"""

losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

model.metrics_names

model.evaluate(test_image_gen)

from tensorflow.keras.preprocessing import image

# https://datascience.stackexchange.com/questions/13894/how-to-get-predictions-with-predict-generator-on-streaming-test-data-in-keras
pred_probabilities = model.predict(test_image_gen)

pred_probabilities

test_image_gen.classes

predictions = pred_probabilities > 0.5

# Numpy can treat this as True/False for us
predictions

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(test_image_gen.classes,predictions))

confusion_matrix(test_image_gen.classes,predictions)

"""# Bild vorhersagen"""

# Your file path will be different!
para_cell

my_image = image.load_img(para_cell,target_size=image_shape)

my_image

type(my_image)

my_image = image.img_to_array(my_image)

type(my_image)

my_image.shape

my_image = np.expand_dims(my_image, axis=0)

my_image.shape

model.predict(my_image)

train_image_gen.class_indices

test_image_gen.class_indices

"""# Gut gemacht!"""

from google.colab import files
files.download('malaria_detector.keras')