import pickle
import tensorflow as tf
from resnet import ResNet50Model, ResNet50ClassificationModel
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



def plot_results_classification(y_true, y_pred, title, filename):
    """
    Plots results for classification tasks, including a confusion matrix and metrics.

    Args:
    - y_true: True class labels.
    - y_pred: Predicted class labels.
    - title: Title of the plot.
    - filename: Filename to save the plot.
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    precision = class_report['weighted avg']['precision']
    recall = class_report['weighted avg']['recall']
    f1 = class_report['weighted avg']['f1-score']

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_labels = sorted(np.unique(y_true))  # Ensure classes are sorted

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"{title}\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")

    # make directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save and show plot
    plt.savefig(filename)
    plt.show()

    # Print classification metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred))


def plot_training_history_classification(history, filename):
    """
    Plots the training and validation accuracy and loss over epochs with dual y-axes.

    Args:
    - history: History object from model.fit().
    - filename: Filename to save the plot.
    """
    import os
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot accuracy on the left y-axis
    ax1.plot(history.history['categorical_accuracy'], label='Training Accuracy', linestyle='-')
    ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy', linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Add a second y-axis for loss
    ax2 = ax1.twinx()
    ax2.plot(history.history['loss'], label='Training Loss', linestyle='-', color='orange')
    ax2.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', color='red')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title and grid
    fig.suptitle('Training and Validation Metrics')
    ax1.grid()

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save and show the plot
    plt.savefig(filename)
    plt.show()












def RN50_regression():
    # with open('../data/data.p', 'rb') as f:
    with open('../data/data_Images_all.p', 'rb') as f:
        data = pickle.load(f)

    images_train = data['train_images']
    images_test = data['test_images']
    # labels_train = data['train_days_remain'].astype('float32')
    # labels_test = data['test_days_remain'].astype('float32')
    labels_train = data['train_age'].astype('float32')
    labels_test = data['test_age'].astype('float32')

    model = ResNet50Model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(), r2_score]
    )

    history = model.fit(
        images_train,
        labels_train,
        validation_split=0.2,
        batch_size=32,
        epochs=40,
    )

    test_loss, test_mae, r2 = model.evaluate(images_test, labels_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")
    print(f"Test r2: {r2}")

    # Generate predictions for the last training samples and test set
    train_predictions = model.predict(images_train) 
    test_predictions = model.predict(images_test)

    folderPath = "results/results_AllImages_Resnet50_age_noImageNet/"

    # Plot results for training samples
    plot_results(
        labels_train, 
        train_predictions, 
        title="y_true vs. y_pred for Last Training Samples",
        filename=f"{folderPath}train_results.png"
    )

    # Plot results for test samples
    plot_results(
        labels_test, 
        test_predictions, 
        title="y_true vs. y_pred for Test Samples",
        filename=f"{folderPath}test_results.png"
    )

    plot_training_history_regression(history, filename=f"{folderPath}training_history.png")





def RN50_classification():
    # Load data
    with open('../data/data_Images_all.p', 'rb') as f:
        data = pickle.load(f)

    images_train = data['train_images']
    images_test = data['test_images']

    # Assuming labels are categorical and need to be one-hot encoded for classification
    labels_train = tf.keras.utils.to_categorical(data['train_days_remain_groups'].astype('int32'))
    labels_test = tf.keras.utils.to_categorical(data['test_days_remain_groups'].astype('int32'))
    # labels_train = tf.keras.utils.to_categorical(data['train_age_groups'].astype('int32'))
    # labels_test = tf.keras.utils.to_categorical(data['test_age_groups'].astype('int32'))

    # Define the classification model
    num_classes = labels_train.shape[1]  # Determine number of classes from one-hot encoding
    model = ResNet50ClassificationModel(num_classes)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    # Train the model
    history = model.fit(
        images_train,
        labels_train,
        validation_split=0.2,
        batch_size=32,
        epochs=40,
    )

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(images_test, labels_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions for the training and test sets
    train_predictions = model.predict(images_train)
    test_predictions = model.predict(images_test)

    # Convert predictions from probabilities to class indices
    train_predicted_classes = tf.argmax(train_predictions, axis=1).numpy()
    test_predicted_classes = tf.argmax(test_predictions, axis=1).numpy()

    # Convert true labels back from one-hot encoding to class indices
    train_true_classes = tf.argmax(labels_train, axis=1).numpy()
    test_true_classes = tf.argmax(labels_test, axis=1).numpy()


    # folderPath = "results/results_AllImages_Resnet50_age_groups/"
    folderPath = "results/results_AllImages_Resnet50_daysRemain_groups_3_noImageNet/"


    # Plot results for training samples
    plot_results_classification(
        train_true_classes,
        train_predicted_classes,
        title="y_true vs. y_pred for Last Training Samples",
        filename=f"{folderPath}train_results.png"
    )

    # Plot results for test samples
    plot_results_classification(
        test_true_classes,
        test_predicted_classes,
        title="y_true vs. y_pred for Test Samples",
        filename=f"{folderPath}test_results.png"
    )

    # Plot training and validation accuracy
    plot_training_history_classification(history, filename=f"{folderPath}training_history.png")


def main():
    RN50_regression()
    # RN50_classification()

if __name__ == '__main__':
    main()