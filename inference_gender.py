import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import time

model = tf.keras.models.load_model('gender.h5')


class_lookup = {0:'female', 1:'male'}

def predict_image(img_path, class_lookup):
    img = image.load_img(img_path, target_size=(260,260))
    x = image.img_to_array(img)
    # Reshapecheckpoint
    x = x.reshape((1,) + x.shape)
    # x /= 255.
    result = model.predict([x])[0][0]
    result_verbose = model.predict([x])
    # print(result_verbose)
    predicted_class = class_lookup[np.argmax(result_verbose, axis=1)[0]]
    predicted_probability = result_verbose[0][np.argmax(result_verbose, axis=1)[0]]

    return predicted_class

folder=r'male'
count=0
male_count=0
female_count=0
acc=0
for filename in os.listdir(folder):
    val=predict_image(os.path.join(folder,filename),class_lookup)
    if val=='male':
        male_count+=1
    print(val)
    img =cv2.imread(os.path.join(folder,filename))
    img=cv2.resize(img,(640,640))
    count+=1
    cv2.putText(img, val, (0,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
# cv2.imwrite('result/male/img{}.jpg'.format(count),img)
acc=(male_count/count)*100
print(acc)