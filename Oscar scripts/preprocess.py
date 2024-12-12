import os
import pickle
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_data(data_folder):
    df = pd.read_csv(f'{data_folder}/filenames_mapped_ages.csv')
    image_data = df[df['filename'].str.contains('.png', na=False)]

    # Extract all necessary columns for training and testing
    train_data = image_data[image_data['testing'] == 0].set_index('filename')
    test_data = image_data[image_data['testing'] == 1].set_index('filename')

    image_size = (224, 224)

    def preprocess(path):
        img = Image.open(path).resize(image_size)
        img_array = np.array(img)
        if img_array.shape[-1] != 3:
            img_array = np.stack([img_array] * 3, axis=-1)
        img_array = img_array / 255.0
        return img_array
    
    def process_images(data):
        images = []
        filenames = []
        ages = []
        days_remain = []
        age_groups = []
        days_remain_groups = []
        day_worm_ids = []
        image_types = []

        for image_id, row in tqdm(data.iterrows(), desc="Processing images", total=data.shape[0]):
            img_path = os.path.join("../Images", image_id)
            if os.path.exists(img_path) and not np.isnan(row['age']):
                image = preprocess(img_path)
                images.append(image)
                filenames.append(image_id)
                ages.append(row['age'])
                days_remain.append(row['daysRemain'])
                age_groups.append(row['age_group'])
                days_remain_groups.append(row['daysRemain_group'])
                day_worm_ids.append(row['day_worm_ID'])
                image_types.append(row.get('image_type', None))  # Get 'image_type' if present, else None
            else:
                pass

        return (
            np.array(images),
            filenames,
            np.array(ages),
            np.array(days_remain),
            np.array(age_groups),
            np.array(days_remain_groups),
            np.array(day_worm_ids),
            image_types,
        )

    # Process train and test data
    (
        train_images, train_filenames, train_age, train_days_remain, train_age_groups,
        train_days_remain_groups, train_day_worm_ids, train_image_types
    ) = process_images(train_data)
    
    (
        test_images, test_filenames, test_age, test_days_remain, test_age_groups,
        test_days_remain_groups, test_day_worm_ids, test_image_types
    ) = process_images(test_data)

    return dict({
        'train_images': train_images,
        'test_images': test_images,
        'train_filenames': train_filenames,
        'test_filenames': test_filenames,
        'train_age': train_age,
        'test_age': test_age,
        'train_days_remain': train_days_remain,
        'test_days_remain': test_days_remain,
        'train_age_groups': train_age_groups,
        'test_age_groups': test_age_groups,
        'train_days_remain_groups': train_days_remain_groups,
        'test_days_remain_groups': test_days_remain_groups,
        'train_day_worm_ids': train_day_worm_ids,
        'test_day_worm_ids': test_day_worm_ids,
        'train_image_types': train_image_types,
        'test_image_types': test_image_types,
    })


def create_pickle(data_folder):
    with open(f'{data_folder}/data_Images_all.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)


if __name__ == '__main__':
    data_folder = '../data'
    create_pickle(data_folder)


# import os
# import pickle

# from PIL import Image
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# def load_data(data_folder):
#     df = pd.read_csv(f'{data_folder}/filenames_mapped_ages.csv')
#     image_data = df[df['filename'].str.contains('.png', na=False)]

#     # train_dic = image_data[image_data['testing'] == 0].set_index('filename')['daysRemain'].to_dict()
#     # test_dic = image_data[image_data['testing'] == 1].set_index('filename')['daysRemain'].to_dict()
#     train_dic = image_data[image_data['testing'] == 0].set_index('filename')['age'].to_dict()
#     test_dic = image_data[image_data['testing'] == 1].set_index('filename')['age'].to_dict()

#     image_size = (224, 224)
#     batch_size = 32

#     def preprocess(path):
#         img = Image.open(path).resize(image_size)
#         img_array = np.array(img)
#         if img_array.shape[-1] != 3:
#             img_array = np.stack([img_array]*3, axis=-1)
#         img_array = img_array / 255.0
#         return img_array
    
#     train_images = []
#     train_labels = []

#     for image_id, days in tqdm(train_dic.items(), desc="Processing images"):
#         img_path = os.path.join("../Images", image_id)
#         if os.path.exists(img_path) and not np.isnan(days):
#             image = preprocess(img_path)
#             train_images.append(image)
#             train_labels.append(days)
#         else:
#             pass

#     test_images = []
#     test_labels = []

#     for image_id, days in tqdm(test_dic.items(), desc="Processing images"):
#         img_path = os.path.join("../Images", image_id)
#         if os.path.exists(img_path) and not np.isnan(days):
#             image = preprocess(img_path)
#             test_images.append(image)
#             test_labels.append(days)
#         else:
#             pass

#     train_images = np.array(train_images)
#     test_images = np.array(test_images)
#     train_labels = np.array(train_labels)
#     test_labels = np.array(test_labels)

#     return dict({'train_images': train_images, 'test_images': test_images, 'train_labels': train_labels, 'test_labels': test_labels})


# def create_pickle(data_folder):
#     with open(f'{data_folder}/data_age.p', 'wb') as pickle_file:
#         pickle.dump(load_data(data_folder), pickle_file)


# if __name__ == '__main__':
#     data_folder = '../data'
#     create_pickle(data_folder)