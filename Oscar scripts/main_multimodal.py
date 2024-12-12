import pickle
import tensorflow as tf

from resnet import ResNet50Model, ResNet50ClassificationModel
from model_MotionFeatures import MLPModel, RandomForestModel, ElasticNetModel
from multimodal_model import MultimodalModel

from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import os

def r2_score(y_true, y_pred):
    # Total sum of squares
    ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # Residual sum of squares
    ss_residual = tf.reduce_sum(tf.square(y_true - y_pred))
    # R^2 score
    r2 = 1 - (ss_residual / (ss_total + tf.keras.backend.epsilon()))  # Add epsilon to avoid division by zero
    return r2



def plot_results(y_true, y_pred, title, filename):
    # Calculate metrics
    y_pred = np.ravel(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    # print(y_true, y_pred)
    # print(y_true.shape, y_pred.shape)
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)  # Line of best fit and stats

    # Generate line of best fit
    line_x = np.linspace(min(y_true), max(y_true), 100)
    line_y = slope * line_x + intercept

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Data Points")
    plt.plot(line_x, line_y, 'g-', label=f"Best Fit: y={slope:.2f}x+{intercept:.2f}")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")

    # Add text with metrics
    plt.text(0.05, 0.95, f"R²: {r_squared:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.90, f"MSE: {mse:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.85, f"MAE: {mae:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.80, f"p-value: {p_value:.3e}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


    # Labels and title
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.grid()

    # make directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save and show plot
    plt.savefig(filename)
    plt.show()

def plot_training_history_regression(history, filename):
    """
    Plots the training and validation loss over epochs for regression.

    Args:
    - history: History object from model.fit().
    - filename: Filename to save the plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Add optional metrics like mean absolute error if available
    if 'mean_absolute_error' in history.history:
        plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    if 'val_mean_absolute_error' in history.history:
        plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')

    plt.xlabel('Epochs')
    plt.ylabel('Loss / Metric Value')
    plt.title('Training and Validation Loss / Metrics')
    plt.legend()
    plt.grid()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save and show the plot
    plt.savefig(filename)
    plt.show()




# def RN50_regression():
#     # with open('../data/data.p', 'rb') as f:
#     with open('../data/data_Images_all.p', 'rb') as f:
#         data = pickle.load(f)

#     images_train = data['train_images']
#     images_test = data['test_images']
#     labels_train = data['train_days_remain'].astype('float32')
#     labels_test = data['test_days_remain'].astype('float32')
    

#     model = ResNet50Model()

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         # optimizer=Adam(learning_rate=0.001),
#         loss=tf.keras.losses.MeanSquaredError(),
#         metrics=[tf.keras.metrics.MeanAbsoluteError(), r2_score]
#     )

#     history = model.fit(
#         images_train,
#         labels_train,
#         validation_split=0.2,
#         batch_size=32,
#         epochs=40,
#     )

#     test_loss, test_mae, r2 = model.evaluate(images_test, labels_test)
#     print(f"Test Loss: {test_loss}")
#     print(f"Test MAE: {test_mae}")
#     print(f"Test r2: {r2}")

#     # Generate predictions for the last training samples and test set
#     train_predictions = model.predict(images_train) 
#     test_predictions = model.predict(images_test)

#     folderPath = "results/results_AllImages_Resnet50_daysRemain/"

#     # Plot results for training samples
#     plot_results(
#         labels_train, 
#         train_predictions, 
#         title="y_true vs. y_pred for Last Training Samples",
#         filename=f"{folderPath}train_results.png"
#     )

#     # Plot results for test samples
#     plot_results(
#         labels_test, 
#         test_predictions, 
#         title="y_true vs. y_pred for Test Samples",
#         filename=f"{folderPath}test_results.png"
#     )

#     plot_training_history_regression(history, filename=f"{folderPath}training_history.png")

def process_data(images, image_day_worm_ids, image_types, tabular_features, tabular_day_worm_ids, labels):
    """
    Aligns images, tabular data, and labels based on day_worm_id.

    Args:
    - images: List of images.
    - image_day_worm_ids: List of day_worm_ids for the images.
    - image_types: List of image types (1 to 4).
    - tabular_features: List of tabular data features.
    - tabular_day_worm_ids: List of day_worm_ids for the tabular data.
    - labels: List of labels corresponding to the day_worm_ids.

    Returns:
    - image_input_arrays: List of NumPy arrays for each image type (ordered by `image_types`).
    - tabular_input_array: NumPy array of tabular features.
    - label_array: NumPy array of labels.
    """
    # Group images by day_worm_id and order by image_type
    grouped_images = {}
    for img, worm_id, img_type in zip(images, image_day_worm_ids, image_types):
        if worm_id not in grouped_images:
            grouped_images[worm_id] = [None] * 4  # Placeholder for 4 image types
        grouped_images[worm_id][img_type - 1] = img  # Ensure ordering

    # Map tabular features and labels to day_worm_id
    tabular_inputs = {worm_id: features for worm_id, features in zip(tabular_day_worm_ids, tabular_features)}
    labels_map = {worm_id: label for worm_id, label in zip(image_day_worm_ids, labels)}

    # Align data by day_worm_id
    image_input_lists = [[] for _ in range(4)]  # For 4 ResNet inputs
    tabular_input_list = []
    label_list = []

    for worm_id in tabular_day_worm_ids:  # Use tabular_day_worm_ids as the reference order
        # print(worm_id)
        # print(type(worm_id))
        if (
            worm_id in grouped_images
            and all(item is not None for item in grouped_images[worm_id])  # Safely check for None
            and worm_id in tabular_inputs
        ):
            for i in range(4):
                image_input_lists[i].append(grouped_images[worm_id][i])  # Ordered by image_type
            tabular_input_list.append(tabular_inputs[worm_id])
            label_list.append(labels_map[worm_id])

    # Convert to numpy arrays
    image_input_arrays = [np.array(image_list) for image_list in image_input_lists]
    tabular_input_array = np.array(tabular_input_list)
    label_array = np.array(label_list)

    return image_input_arrays, tabular_input_array, label_array


def prepare_multimodal_inputs(image_data, tabular_data):
    """
    Prepares inputs for the multimodal model by aligning data based on `day_worm_ids`
    and ensuring image inputs are ordered by `image_types`.

    Args:
    - image_data: Dictionary with keys for train and test datasets.
    - tabular_data: Dictionary with keys for train and test datasets.

    Returns:
    - train_image_inputs: List of NumPy arrays for each image type (train set, ordered by `image_types`).
    - test_image_inputs: List of NumPy arrays for each image type (test set, ordered by `image_types`).
    - train_tabular_inputs: NumPy array of tabular features (train set).
    - test_tabular_inputs: NumPy array of tabular features (test set).
    - train_labels: NumPy array of labels (train set).
    - test_labels: NumPy array of labels (test set).
    """


    tabular_data_train = tabular_data['train_features']
    tabular_data_test = tabular_data['test_features']

    # remove the singleton layer in the middle (otherwise doesn't train at all, also won't give error)
    tabular_data_train = tabular_data_train.reshape(tabular_data_train.shape[0], -1)  # (347, 2924)
    tabular_data_test = tabular_data_test.reshape(tabular_data_test.shape[0], -1)

    # Process training data
    train_images, train_tabular_inputs, train_labels = process_data(
        images=image_data['train_images'],
        image_day_worm_ids=image_data['train_day_worm_ids'],
        image_types=image_data['train_image_types'],
        tabular_features=tabular_data_train,
        tabular_day_worm_ids=tabular_data['train_day_worm_ids'],
        labels=image_data['train_age']
    )

    # Process testing data
    test_images, test_tabular_inputs, test_labels = process_data(
        images=image_data['test_images'],
        image_day_worm_ids=image_data['test_day_worm_ids'],
        image_types=image_data['test_image_types'],
        tabular_features=tabular_data_test,
        tabular_day_worm_ids=tabular_data['test_day_worm_ids'],
        labels=image_data['test_age']
    )

    return train_images, test_images, train_tabular_inputs, test_tabular_inputs, train_labels, test_labels


def multimodal_ImageNFeature():
    # Load datasets
    with open('../data/data_Images_all.p', 'rb') as f:
        image_data = pickle.load(f)
    with open('../data/data_motionFeatures_all.p', 'rb') as f:
        tabular_data = pickle.load(f)

    # Prepare inputs
    train_images, test_images, train_tabular_inputs, test_tabular_inputs, train_labels, test_labels = prepare_multimodal_inputs(image_data, tabular_data)

    # Initialize ResNet models
    resnet_models = [ResNet50Model() for _ in range(4)]

    # Initialize MLP model
    mlp_model = MLPModel(input_dim=train_tabular_inputs.shape[-1])

    # Initialize the multimodal model
    multimodal_model = MultimodalModel(resnet_models=resnet_models, mlp_model=mlp_model)

    # Compile the multimodal model
    multimodal_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"]
    )

    # Train the model
    history = multimodal_model.fit(
        [train_images, train_tabular_inputs],
        train_labels,
        validation_split=0.2,
        batch_size=32,
        epochs=40,
    )

    # Evaluate the model
    test_loss, test_mae , test_mse = multimodal_model.evaluate([test_images, test_tabular_inputs], test_labels)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")
    print(f"Test MSE: {test_mse}")
    


    # Generate predictions
    train_predictions = multimodal_model.predict([train_images, train_tabular_inputs])
    test_predictions = multimodal_model.predict([test_images, test_tabular_inputs])

    # Plot results
    folderPath = "results/results_multimodal_age_ImageNFeatureOnly_1/"

    plot_results(train_labels, train_predictions, title="Multimodal: y_true vs. y_pred (Training)", filename=f"{folderPath}train_results.png")
    plot_results(test_labels, test_predictions, title="Multimodal: y_true vs. y_pred (Testing)", filename=f"{folderPath}test_results.png")

    # Plot training history
    plot_training_history_regression(history, filename=f"{folderPath}training_history.png")



def main():
    multimodal_ImageNFeature()

if __name__ == '__main__':
    main()