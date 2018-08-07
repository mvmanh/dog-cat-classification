from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import os
import numpy as np

# dimensions of our images.
img_width, img_height = 150, 150
train_data_dir = '/home/mvmanh/dataset/PetImages/train'
validation_data_dir = '/home/mvmanh/dataset/PetImages/validation'
test_data_dir = '/home/mvmanh/dataset/PetImages/test'
nb_train_samples = 17500
nb_validation_samples = 2500
nb_test_samples = 5000
epochs = 30
batch_size = 32
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
model = None
if not os.path.exists('dog-cat-final-model.h5'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='cat-dog-classification.png', show_shapes=True)
    if os.path.exists('dog-cat-weights.h5'):
        model.load_weights('dog-cat-weights.h5')
    else:
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            verbose=1)
        model.save_weights('dog-cat-weights.h5')
    model.save('dog-cat-final-model.h5')
else:
    print('Load entire model from file')
    model = load_model('dog-cat-final-model.h5')
print('Start evaluating')
score = model.evaluate_generator(test_generator, steps= nb_test_samples // batch_size)
print('Evaluate loss: {0}, accuracy {1}'.format(score[0], score[1]))


image = image.load_img('dataset/PetImages/test/Dog/6500.jpg',target_size=(150,150,3))
image = np.asarray(image)
image = np.expand_dims(image, axis=0)
image = image * 1.0 / 255

predicted = model.predict(image)
print('Predicted result: {0} '.format(predicted))
