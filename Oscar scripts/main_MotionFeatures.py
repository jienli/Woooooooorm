import pickle
import tensorflow as tf
from model_MotionFeatures import MLPModel, RandomForestModel, ElasticNetModel
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

import os





# def plot_results(y_true, y_pred, title, filename):
#     plt.figure(figsize=(8, 8))
#     plt.scatter(y_true, y_pred, alpha=0.7)
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
#     plt.xlabel("True Values")
#     plt.ylabel("Predicted Values")
#     plt.title(title)
#     plt.legend()
#     plt.grid()
#     plt.savefig(filename)
#     plt.show()



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



def main():
    with open('../data/data_motionFeatures_all.p', 'rb') as f:
        data = pickle.load(f)

    images_train = data['train_features']
    images_test = data['test_features']
    # labels_train = data['train_age'].astype('float32')
    # labels_test = data['test_age'].astype('float32')
    labels_train = data['train_days_remain'].astype('float32')
    labels_test = data['test_days_remain'].astype('float32')

    # remove the singleton layer in the middle (otherwise doesn't train at all, also won't give error)
    images_train = images_train.reshape(images_train.shape[0], -1)  # (347, 2924)
    images_test = images_test.reshape(images_test.shape[0], -1)
    
    # Standardize the data for Elastic Net
    # scaler = StandardScaler()
    # images_train = scaler.fit_transform(images_train)
    # images_test = scaler.transform(images_test)

    print(f"Training data shape: {images_train.shape}")
    print(f"Testing data shape: {images_test.shape}")
    
    # ----- MLP Model -----
    print("=== MLP Model ===")
    input_dim = images_train.shape[-1]
    model = MLPModel(input_dim=input_dim)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",  # Mean Squared Error for regression
                  metrics=["mae", "mse"])  # Mean Absolute Error for evaluation

    model.fit(
        images_train,
        labels_train,
        validation_split=0.2,
        batch_size=32,
        epochs=50,
    )

    test_loss, test_mae, test_mse = model.evaluate(images_test, labels_test)
    print(f"MLP Test Loss: {test_loss}")
    print(f"MLP Test MAE: {test_mae}")
    print(f"MLP Test MSE: {test_mae}")

    # Plot results for MLP
    train_predictions = model.predict(images_train)
    test_predictions = model.predict(images_test)


    folderPath = "results/results_motionFeatures_daysRemain2/"
    plot_results(labels_train, train_predictions, title="MLP: y_true vs. y_pred (Training)", filename=f"{folderPath}mlp_train_results.png")
    plot_results(labels_test, test_predictions, title="MLP: y_true vs. y_pred (Testing)", filename=f"{folderPath}mlp_test_results.png")

    # these 2 models performed worse than MLP, so ignore
    # # ----- Random Forest Model -----
    # print("\n=== Random Forest ===")
    # rf_model = RandomForestModel(n_estimators=100, max_depth=10)
    # rf_model.fit(images_train, labels_train)
    # y_pred_rf = rf_model.predict(images_test)

    # print("Random Forest MSE:", mean_squared_error(labels_test, y_pred_rf))
    # print("Random Forest R^2:", r2_score(labels_test, y_pred_rf))


    # plot_results(labels_test, y_pred_rf, title="Random Forest: y_true vs. y_pred (Testing)", filename="rf_test_results.png")

    # # ----- Elastic Net Model -----
    # print("\n=== Elastic Net ===")
    # en_model = ElasticNetModel(input_dim=images_train.shape[-1], alpha=0.1, l1_ratio=0.5)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # for epoch in range(20):
    #     with tf.GradientTape() as tape:
    #         y_pred_en = en_model(images_train)
    #         loss = en_model.compute_loss(labels_train, y_pred_en)
    #     gradients = tape.gradient(loss, en_model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, en_model.trainable_variables))
    #     print(f"Epoch {epoch+1}, Elastic Net Loss: {loss.numpy()}")

    # y_pred_en_test = en_model(images_test).numpy()
    # print("Elastic Net MSE:", mean_squared_error(labels_test, y_pred_en_test))
    # print("Elastic Net R^2:", r2_score(labels_test, y_pred_en_test))

    # plot_results(labels_test, y_pred_en_test, title="Elastic Net: y_true vs. y_pred (Testing)", filename="en_test_results.png")


if __name__ == '__main__':
    main()







# def r2_score(y_true, y_pred):
#     # Total sum of squares
#     ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
#     # Residual sum of squares
#     ss_residual = tf.reduce_sum(tf.square(y_true - y_pred))
#     # R^2 score
#     r2 = 1 - (ss_residual / (ss_total + tf.keras.backend.epsilon()))  # Add epsilon to avoid division by zero
#     return r2


# def plot_results(y_true, y_pred, title, filename):
#     plt.figure(figsize=(8, 8))
#     plt.scatter(y_true, y_pred, alpha=0.7)
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
#     plt.xlabel("True Values")
#     plt.ylabel("Predicted Values")
#     plt.title(title)
#     plt.legend()
#     plt.grid()
#     plt.savefig(filename)
#     plt.show()


# def main():
#     with open('../data/data_motionFeatures.p', 'rb') as f:
#         data = pickle.load(f)

#     images_train = data['train_features']
#     images_test = data['test_features']
#     labels_train = data['train_labels'].astype('float32')
#     labels_test = data['test_labels'].astype('float32')
#     print(images_train.shape)
#     print(images_test.shape)
    
#     # Save the DataFrame as a CSV file
#     # images_train_df = pd.DataFrame(images_train.reshape(images_train.shape[0], -1))
#     # images_train_df.to_csv("images_train_debug.csv", index=False)

#     # Input dimension (number of features)
#     input_dim = images_train.shape[-1]

#     # Instantiate the model
#     model = MLPModel(input_dim=input_dim)

#     # Compile the model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                 loss="mse",  # Mean Squared Error for regression
#                 metrics=["mae"])  # Mean Absolute Error for evaluation



#     model.fit(
#         images_train,
#         labels_train,
#         validation_split=0.2,
#         batch_size=32,
#         epochs=100,
#     )

#     test_loss, test_mae = model.evaluate(images_test, labels_test)
#     print(f"Test Loss: {test_loss}")
#     print(f"Test MAE: {test_mae}")
#     # print(f"Test r2: {r2}")

#     # Generate predictions for the last training samples and test set
#     train_predictions = model.predict(images_train) 
#     test_predictions = model.predict(images_test)

#     # Plot results for training samples
#     plot_results(
#         labels_train, 
#         train_predictions, 
#         title="y_true vs. y_pred for Last Training Samples",
#         filename="train_results.png"
#     )

#     # Plot results for test samples
#     plot_results(
#         labels_test, 
#         test_predictions, 
#         title="y_true vs. y_pred for Test Samples",
#         filename="test_results.png"
#     )


# if __name__ == '__main__':
#     main()