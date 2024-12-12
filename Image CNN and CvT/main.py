import pickle
import os
import numpy as np
import tensorflow as tf
from cnn import CNNModel
from cvt import CvTModel

def get_worm_data():
    with open('../data/data.p', 'rb') as f:
        data = pickle.load(f)

    return data['train_images'], data['test_images'], data['train_labels'], data['test_labels']

def r2_score(y_true, y_pred):
    ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    ss_residual = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - (ss_residual / (ss_total + tf.keras.backend.epsilon()))  
    return r2

def save_results(save_dir, model, y_true, y_pred):
    os.makedirs(save_dir, exist_ok=True)

    weights_path = os.path.join(save_dir, 'weights.h5')
    predictions_path = os.path.join(save_dir, 'predictions.npy')
    
    model.save_weights(weights_path)
    np.save(predictions_path, {'y_true': y_true, 'y_pred': y_pred})

def main():
    images_train, images_test, labels_train, labels_test = get_worm_data()

    num_classes = 1
    #model = CNNModel(num_classes)
    model = CvTModel(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(), r2_score]
    )

    model.fit(
        images_train,
        labels_train,
        validation_split=0.2,
        batch_size=16,
        epochs=40,
    )

    test_loss, test_mae, r2 = model.evaluate(images_test, labels_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")
    print(f"Test r2: {r2}")

    y_pred = model.predict(images_test)

    if isinstance(model, CvTModel):
        save_results('./results/CvTModel', model, labels_test, y_pred)
    elif isinstance(model, CNNModel):
        save_results('./results/CNNModel', model, labels_test, y_pred)
    else:
        raise TypeError

if __name__ == '__main__':
    main()