import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.applications import VGG16
import matplotlib.image as mpimg
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input

# load the VGG16 network *pre-trained* on the ImageNet dataset
model = VGG16(weights="imagenet")
model.summary()

def load_and_process_image(image_path):
    # Load in the image with a target size of 224,224
    image = image_utils.load_img(image_path, target_size=(224, 224))
    # Convert the image from a PIL format to a numpy array
    image = image_utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1,224,224,3)
    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)
    # Print image's shape after processing
    print('\r\nimage shape from %s to %s' % (mpimg.imread(image_path).shape, image.shape))
    return image

def readable_prediction(image_path):
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))
import numpy as np
 
def doggy_door(image_path):
    image = load_and_process_image(image_path)
    preds = model.predict(image)
    if 151 <= np.argmax(preds) <= 268:
        print("Doggy come on in!")
    elif 281 <= np.argmax(preds) <= 285:
        print("Kitty stay inside!")
    else:
        print("You're not a dog! Stay outside!")

readable_prediction("pictures/sleepy_cat.jpg")
readable_prediction("pictures/happy_dog.jpg")
readable_prediction("pictures/hashiqi.jpg")
readable_prediction("pictures/dog2.jpg")

doggy_door("pictures/sleepy_cat.jpg")
doggy_door("pictures/happy_dog.jpg")
doggy_door("pictures/hashiqi.jpg")
doggy_door("pictures/dog2.jpg")