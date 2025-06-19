# import pickle

# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# import numpy as np
# from numpy.linalg import norm
# import os
# from tqdm import tqdm

# model= ResNet50(weights='imagenet' ,include_top=False, input_shape=(224,224,3))
# model.trainable=False

# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# # print(model.summary())

# def extract_features(img_path,model):
#     img =image.load_img(img_path, target_size=(224,224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result=model.predict(preprocessed_img).flatten()
#     normalized_result=result / norm(result)
#     return normalized_result

# filenames=[]
# for file in os.listdir('images'):
#  filenames.append(os.path.join('images',file))

# feature_list=[]

# for file in tqdm(filenames):
#     feature_list.append(extract_features(file,model))

# pickle.dump(feature_list,open('embeddings.pkl','wb'))
# pickle.dump(filenames,open('filenames.pkl','wb'))
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Load Cloudinary URLs
with open("cloudinary_urls.pkl", "rb") as f:
    filenames = pickle.load(f)

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Feature extraction function from URL
def extract_features(img_url, model):
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Failed for {img_url}: {e}")
        return None

# Extract features from all URLs
feature_list = []
for url in tqdm(filenames):
    features = extract_features(url, model)
    if features is not None:
        feature_list.append(features)
    else:
        # Optionally handle errors gracefully
        feature_list.append(np.zeros(2048))  # or skip

# Save embeddings and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
