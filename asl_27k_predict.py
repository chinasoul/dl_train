from tensorflow import keras
import numpy as np

model = keras.models.load_model('asl_27k')
model.summary()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)

show_image('asl_27k/b.png')

from tensorflow.keras.preprocessing import image as image_utils

def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    return image

alphabet = "abcdefghijklmnopqrstuvwxyz"
dictionary = {}
for i in range(26):
    dictionary[i] = alphabet[i]

def predict_letter(file_path):
    show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1) 
    image = image/255
    prediction = model.predict(image)
    # convert prediction to letter
    predicted_letter = dictionary[np.argmax(prediction)]
    print(predicted_letter)
    return predicted_letter

predict_letter("asl_27k/a.png")
predict_letter("asl_27k/b.png")