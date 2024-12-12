import os
import pickle
import re
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_data(data_folder):
    # Read tierpsy features summary file
    df = pd.read_csv(f'{data_folder}/../Videos_60s_all_temp/filenames_summary_tierpsy_plate_20241205_033129.csv', skiprows=4)
    df = df[df['is_good'] == True]

    # Extract strings from the "filename" column
    def extract_strings(filename):
        last_part = filename.split("/")[-1]
        day_match = re.search(r"(Day\d+)", last_part)
        video_match = re.search(r"Video-(\d+)_featuresN", last_part)
        return (day_match.group(1) if day_match else None, video_match.group(1) if video_match else None)

    df['extracted'] = df['filename'].apply(extract_strings)
    result_dict = df.set_index('file_id')['extracted'].to_dict()

    # Read metadata
    df_meta = pd.read_csv(f'{data_folder}/filenames_mapped_ages.csv')

    # Function to match metadata based on filename patterns
    def match_meta(row, mapped_df, col):
        day, number = row
        matched_row = mapped_df[mapped_df['filename'].str.contains(fr"{day}_.*{number}.*", regex=True)]
        return matched_row[col].iloc[0] if not matched_row.empty else None

    # Map metadata
    mapped_values = {
        key: {
            "age": match_meta(value, df_meta, "age"),
            "age_group": match_meta(value, df_meta, "age_group"),
            "daysRemain": match_meta(value, df_meta, "daysRemain"),
            "daysRemain_group": match_meta(value, df_meta, "daysRemain_group"),
            "day_worm_ID": match_meta(value, df_meta, "day_worm_ID"),
        }
        for key, value in result_dict.items()
    }

    # Split into training and testing dictionaries
    testing_keys = df_meta[df_meta['testing'] == 1].index
    training_dict = {k: v for k, v in mapped_values.items() if k not in testing_keys}
    testing_dict = {k: v for k, v in mapped_values.items() if k in testing_keys}

    # Read features and normalize
    df_features = pd.read_csv(f'{data_folder}/../Videos_60s_all_temp/features_summary_tierpsy_plate_20241205_033129.csv', skiprows=1)
    df_features = df_features.apply(lambda x: x.fillna(x.mean()), axis=0).dropna(axis=1, how='all')

    # Normalize features
    first_column = df_features.iloc[:, 0]
    df_numeric = df_features.iloc[:, 1:].copy()
    df_numeric = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
    df_features = pd.concat([first_column, df_numeric], axis=1)

    # Process train and test data
    def process_features(data_dict, df_features):
        features = []
        ages = []
        age_groups = []
        days_remain = []
        days_remain_groups = []
        day_worm_ids = []

        for feature_id, meta in tqdm(data_dict.items(), desc="Processing features"):
            if meta["age"] is not None and not np.isnan(meta["age"]):
                feature_row = df_features[df_features['file_id'] == feature_id].iloc[:, 1:]
                features.append(feature_row)
                ages.append(meta["age"])
                age_groups.append(meta["age_group"])
                days_remain.append(meta["daysRemain"])
                days_remain_groups.append(meta["daysRemain_group"])
                day_worm_ids.append(meta["day_worm_ID"])
            else:
                pass

        return (
            np.array(features),
            np.array(ages),
            np.array(age_groups),
            np.array(days_remain),
            np.array(days_remain_groups),
            day_worm_ids,
        )

    # Process training and testing features
    train_features, train_age, train_age_groups, train_days_remain, train_days_remain_groups, train_day_worm_ids = process_features(
        training_dict, df_features
    )
    test_features, test_age, test_age_groups, test_days_remain, test_days_remain_groups, test_day_worm_ids = process_features(
        testing_dict, df_features
    )

    return dict({
        'train_features': train_features,
        'test_features': test_features,
        'train_age': train_age,
        'test_age': test_age,
        'train_age_groups': train_age_groups,
        'test_age_groups': test_age_groups,
        'train_days_remain': train_days_remain,
        'test_days_remain': test_days_remain,
        'train_days_remain_groups': train_days_remain_groups,
        'test_days_remain_groups': test_days_remain_groups,
        'train_day_worm_ids': train_day_worm_ids,
        'test_day_worm_ids': test_day_worm_ids,
    })

def create_pickle(data_folder):
    print(load_data(data_folder))
    with open(f'{data_folder}/data_motionFeatures_all.p', 'wb') as pickle_file:
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

# import re

# def load_data(data_folder):

#     # read tierpsy features summary file (with file id to filename(which contains worm id))
#     df = pd.read_csv(f'{data_folder}/../Videos_60s_all_temp/filenames_summary_tierpsy_plate_20241205_033129.csv', skiprows = 4)
#     df = df[df['is_good'] == True]

#     # Define a function to extract the two required strings from the "filename" column
#     def extract_strings(filename):
#         # Extract the last part of the path
#         last_part = filename.split("/")[-1]
#         # Find "DayXX" and the number after "Video-" but before "_featuresN"
#         day_match = re.search(r"(Day\d+)", last_part)
#         video_match = re.search(r"Video-(\d+)_featuresN", last_part)
#         return (day_match.group(1) if day_match else None, video_match.group(1) if video_match else None)

#     # Apply the function to the "filename" column
#     df['extracted'] = df['filename'].apply(extract_strings)

#     # Create the dictionary using "file_id" as keys and the extracted strings as values
#     result_dict = df.set_index('file_id')['extracted'].to_dict()

#     # print(result_dict)

#     df_meta = pd.read_csv(f'{data_folder}/filenames_mapped_ages.csv')

#     # Function to match "daysRemain" based on filename patterns
#     def match_days_remain(row, mapped_df):
#         day, number = row
#         matched_row = mapped_df[mapped_df['filename'].str.contains(fr"{day}_.*{number}.*", regex=True)]
#         # return matched_row['daysRemain'].iloc[0] if not matched_row.empty else None
#         return matched_row['age'].iloc[0] if not matched_row.empty else None

#     # Map dictionary values to "daysRemain"
#     mapped_values = {key: match_days_remain(value, df_meta) for key, value in result_dict.items()}

#     # Split into training and testing dictionaries based on the "testing" column
#     testing_keys = df_meta[df_meta['testing'] == 1].index
#     training_dict = {k: v for k, v in mapped_values.items() if k not in testing_keys}
#     testing_dict = {k: v for k, v in mapped_values.items() if k in testing_keys}

#     # print(training_dict)
#     # print("================================")
#     # print(testing_dict)


#     df_features = pd.read_csv(f'{data_folder}/../Videos_60s_all_temp/features_summary_tierpsy_plate_20241205_033129.csv', skiprows=1)

#     # interpolate nan feature values with column mean (to make sure it's not biasing to nan values)
#     df_features = df_features.apply(lambda x: x.fillna(x.mean()), axis=0)
#     # remove empty columns
#     df_features = df_features.dropna(axis=1, how='all')



#     # normalize (min max)

#     # Separate the first column
#     first_column = df_features.iloc[:, 0]  # Extract the first column

#     # Numeric part of the DataFrame (excluding the first column)
#     df_numeric = df_features.iloc[:, 1:].copy()

#     # Normalize the numeric part
#     df_numeric = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

#     # Reassemble the DataFrame
#     df_features = pd.concat([first_column, df_numeric], axis=1)  # Combine the first column and normalized data

#     # Save the normalized DataFrame to CSV
#     # output_path = "df_features_normalized.csv"
#     # df_features.to_csv(output_path, index=False)





#     train_features = []
#     train_labels = []
#     for feature_id, days in tqdm(training_dict.items(), desc = "processing training features"):
#         # print(feature_id, days)
#         if days != None and not np.isnan(days):
#             train_features.append(df_features[df_features['file_id'] == feature_id].iloc[:, 1:])
#             train_labels.append(days)
#         else:
#             pass

#     test_features = []
#     test_labels = []
#     for feature_id, days in tqdm(testing_dict.items(), desc = "processing testing features"):
#         if days != None and not np.isnan(days):
#             test_features.append(df_features[df_features['file_id'] == feature_id].iloc[:, 1:])
#             test_labels.append(days)
#         else:
#             pass


#     train_features = np.array(train_features)
#     test_features = np.array(test_features)
#     train_labels = np.array(train_labels)
#     test_labels = np.array(test_labels)


#     return dict({'train_features': train_features, 'test_features': test_features, 'train_labels': train_labels, 'test_labels': test_labels})



# def create_pickle(data_folder):
#     with open(f'{data_folder}/data_motionFeatures_age.p', 'wb') as pickle_file:
#         pickle.dump(load_data(data_folder), pickle_file)


# if __name__ == '__main__':
#     data_folder = '../data'
#     # print( load_data(data_folder))
#     create_pickle(data_folder)