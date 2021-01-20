import pandas as pd
import tensorflow
import sys

# remember to export LD_LIBRARY_PATH
if not len(tensorflow.config.experimental.list_physical_devices('GPU')):
    print("not using GPU, quiting...")
    sys.exit(-1)
else:
    pass

# dataset from https://www.kaggle.com/datamunge/sign-language-mnist
# no J&Z but label is 0~24(w/o 9)
train_df = pd.read_csv("asl_27k/sign_mnist_train.csv")
test_df = pd.read_csv("asl_27k/sign_mnist_test.csv")

# print(train_df.head())
# print(test_df.head())

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']
x_train = train_df.values
x_test = test_df.values

print(x_train.shape)
print(y_train.shape)

x_train = x_train/255
x_test = x_test/255

# Reshape the image data for the convolutional network
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape)
print(y_train.shape)

#https://www.reddit.com/r/MLQuestions/comments/7ycjca/confused_about_using_to_categorical_in/
#https://github.com/rstudio/keras/issues/53
num_classes = 26
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))

model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image 
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images horizontally
#         vertical_flip=False)  # Don't randomly flip images vertically
# datagen.fit(x_train)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(datagen.flow(x_train,y_train, batch_size=32), # Default batch_size is 32. We set it here for clarity.
#           epochs=10,
#           steps_per_epoch=len(x_train)/32, # Run same number of steps we would if we were not using a generator.
#           validation_data=(x_test, y_test))
model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test))
model.save('asl_27k')
