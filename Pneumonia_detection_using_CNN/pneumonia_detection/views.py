from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the correct relative path to the model
# model_path = os.path.join(os.getcwd(), 'pneumonia_cnn_model.h5')
model_path = r'C:\Users\umayy\MCA_Main_Project\pneumonia_cnn_model.h5'
model = load_model(model_path)

def index(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        img_path = os.path.join('media', img.name)
        # img_path = os.path.join(settings.MEDIA_ROOT, img.name)
        with open(img_path, 'wb') as f:
            for chunk in img.chunks():
                f.write(chunk)

        test_image = tf.keras.utils.load_img(img_path, target_size=(256, 256))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        class_probabilities = result[0]
        if class_probabilities[0] > class_probabilities[1]:
            result = "Normal"
        else:
            result = "Pneumonia"
        return render(request, 'pneumonia_detection/index.html', {'result': result, 'image': img_path})
    return render(request, 'pneumonia_detection/index.html')

