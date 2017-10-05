from keras.layers import Conv2D, MaxPool2D, Dense, Activation, Dropout, Flatten
import numpy as np
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import Sequential
from keras import backend as K

batch_size = 16
# for i in range(1,100): try: img = Image.open(
# "/home/raghav/Dropbox/coding/python/google_dinosaur_ANN/train_data/jump_images/testimage{}.jpg".format(i)) except
# Exception: continue width, height = img.size print((width, height))
#
if K.image_data_format() == 'channels_first':
    input_shape = (3, 300, 300)
else:
    input_shape = (300, 300, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add((MaxPool2D(pool_size=(2, 2))))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, shear_range=0.2,)

train_generator = train_datagen.flow_from_directory(directory="/home/raghav/Dropbox/coding/python/GoogleDinosaur_CNN/"
                                                              "train_data", target_size=(300, 300), class_mode='binary')

#model.fit_generator(train_generator, epochs=2, steps_per_epoch=200 // batch_size)

#model.save_weights('first_try.h5')
model.load_weights('first_try.h5')
img = load_img("/home/raghav/Dropbox/coding/python/GoogleDinosaur_CNN/test_data/no_jump_testimage1.jpg")
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict_classes(x)
prob = model.predict_proba(x)
print(preds, prob)
if preds == 0:
    print("jump")
else:
    print("no jump")
