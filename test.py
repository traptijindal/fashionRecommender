# import pickle
# import numpy as np
# import tensorflow
# from numpy.linalg import norm
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from sklearn.neighbors import NearestNeighbors
# import cv2

# feature_list = np.array( pickle.load(open('embeddings.pkl', 'rb')))

# filenames = pickle.load(open('filenames.pkl','rb'))

# model= ResNet50(weights='imagenet' ,include_top=False, input_shape=(224,224,3))
# model.trainable=False

# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# img =image.load_img('sample/saree.webp', target_size=(224,224))
# img_array = image.img_to_array(img)
# expanded_img_array = np.expand_dims(img_array, axis=0)
# preprocessed_img = preprocess_input(expanded_img_array)
# result=model.predict(preprocessed_img).flatten()
# normalized_result=result / norm(result)\


# neighbours=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
# neighbours.fit(feature_list)

# distances,indices= neighbours.kneighbors([normalized_result])

# print(indices)

# for file in indices[0]:
#     temp_img = cv2.imread(filenames[file])
#     resized_img = cv2.resize(temp_img, (512, 512))
#     cv2.imshow('output', resized_img)
#     cv2.waitKey(0)
      
import pickle
import numpy as np
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import requests
from PIL import Image
from io import BytesIO

# Load data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load and preprocess query image
img = image.load_img('sample/saree.webp', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Nearest neighbors search
neighbours = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbours.fit(feature_list)
distances, indices = neighbours.kneighbors([normalized_result])

# Display results using PIL
for file in indices[0]:
    url = filenames[file]
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((512, 512))
    img.show()  # This opens the image in your default viewer
