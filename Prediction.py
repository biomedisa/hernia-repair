import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def hernien_höhe(path_to_model,path_to_data):

    model = tf.keras.models.load_model(path_to_model)

    datagen = ImageDataGenerator( 
    )

    ds_test = datagen.flow_from_directory(
        path_to_data,
        class_mode= None,
        color_mode='rgb',
        batch_size= 32,
        shuffle= False,
        target_size=(512,512),
    )


    prediction = model.predict(ds_test, verbose=0)
    smooth_prediction = np.empty_like(prediction)
    for slice in range(prediction.size):
        smooth_prediction[slice] = sum(prediction[(i+slice)%prediction.size] for i in range(-10,11)) * 1./21

    hernien_interval = np.argwhere(smooth_prediction>=0.5)

    hernien_height = hernien_interval[-1][0]-hernien_interval[0][0]

    return hernien_height