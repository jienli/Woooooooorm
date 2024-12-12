import os
import pickle

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

def preprocess(image_ids, data_folder, input_shape):
    image_features = []

    for image_id in tqdm(image_ids, desc='Extracting Features'):
        path = os.path.join(data_folder, image_id)
        img = Image.open(path).resize(input_shape)
        img_array = np.array(img) / 255.0
        if img_array.shape[-1] != 3:
            img_array = np.stack([img_array] * 3, axis=-1)
        image_features.append(img_array)

    return image_features

def load_data(data_folder):
    df = pd.read_csv(f'{data_folder}/filenames_mapped_ages.csv')
    image_data = df[df['filename'].str.contains('.png', na=False)].copy()

    image_data = image_data.dropna(subset=['age', 'testing'])   
    # image_data.loc[:, 'lifestage'] = pd.qcut(image_data['age'], q=5, labels=[0, 1, 2, 3, 4])

    train_data = image_data[image_data['testing'] == 0]
    test_data = image_data[image_data['testing'] == 1]

    train_images = train_data['filename'].tolist()
    train_labels = train_data['age'].tolist()
    test_images = test_data['filename'].tolist()
    test_labels = test_data['age'].tolist()
    
    image_folder = "../data/Images"
    input_shape = (128, 128)
    train_images = preprocess(train_images, image_folder, input_shape)
    test_images = preprocess(test_images, image_folder, input_shape)
    
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return dict({'train_images': train_images, 'test_images': test_images, 'train_labels': train_labels, 'test_labels': test_labels})

def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)

if __name__ == '__main__':
    data_folder = '../data'
    create_pickle(data_folder)