import pickle
import pandas as pd
import numpy as np
import cv2

# Read the dataframe
with open('data_eye_cordinates.pickle','rb') as f:
    data=pickle.load(f)


# Know more about your data
print('Columns: ', data.columns)
print('Shape of each frame/Image: ', data.iloc[0][1].shape)
print('shape of right/left eye cordinates: ', data.iloc[0][2].shape)
print('Total number of frames: ', data.shape[0])

# Reconstruct image from the image/pixel values
image = data.iloc[0][1]
cv2.imwrite('test_image.jpg',image)
