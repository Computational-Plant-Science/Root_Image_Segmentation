import preProcess
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model, load_model
import cv2
import numpy as np
from PIL import ImageOps
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
from PIL import Image
import skimage.transform as trans
import skimage.io as io
from skimage.transform import resize

testData = preProcess.testData("/root_data2/test3/")
new_model = load_model('/model_root_test.h5')
# results = model.predict_generator(testGene,30,verbose=1)
results = new_model.predict(testData,5,verbose=1)
resize_image = resize(results[0], (3456, 5184))
plt.imshow(resize_image,aspect="auto",cmap='gray')
print("Done")

# for i in range(0,30):
#     img = load_img("Z:/Works Kabir/Root segmentations/unet-master/backup/root_data2/test3_label/"+str(i)+".tiff")
#     img2=np.array(img)
#     img2[img2==1]=0
#     img2[img2==2]=255
#     img2 = Image.fromarray(img2)
#     #img2.save("Z:/Works Kabir/Root segmentations/unet-master/backup/root_data2/label2/"+str(i)+".png")
#
# print("Done")



