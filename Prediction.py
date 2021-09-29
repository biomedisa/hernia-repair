import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_hernia_length(path_to_model,path_to_data):

    model = tf.keras.models.load_model(path_to_model)

    datagen = ImageDataGenerator( 
    )

    hernia_data = datagen.flow_from_directory(
        path_to_data,
        class_mode= None,
        color_mode='rgb',
        batch_size= 32,
        shuffle= False,
        target_size=(512,512),
    )

    prediction = model.predict(hernia_data, verbose=0)
    smooth_prediction = np.empty_like(prediction)
    for slice in range(prediction.size):
        smooth_prediction[slice] = sum(prediction[(i+slice)%prediction.size] for i in range(-10,11)) * 1./21

    hernia_interval = np.argwhere(smooth_prediction>=0.5)
    hernia_length = hernia_interval[-1][0]-hernia_interval[0][1]

    return hernia_length