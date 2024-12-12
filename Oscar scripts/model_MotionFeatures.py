
import tensorflow as tf

from sklearn.tree import DecisionTreeRegressor
import numpy as np

class MLPModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        
        # Define the layers
        self.dense1 = tf.keras.layers.Dense(units=512, activation="relu")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        
        self.dense2 = tf.keras.layers.Dense(units=256, activation="relu")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        
        self.dense3 = tf.keras.layers.Dense(units=128, activation="relu")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        
        self.final_dense = tf.keras.layers.Dense(units=1, activation="linear")  # Regression output
        
    def call(self, inputs, training=False):
        # Forward pass
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.batch_norm3(x, training=training)
        x = self.dropout3(x, training=training)
        
        output = self.final_dense(x)
        return output


class RandomForestModel(tf.keras.Model):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super(RandomForestModel, self).__init__()
        self.n_estimators = n_estimators
        self.models = [
            DecisionTreeRegressor(max_depth=max_depth, random_state=random_state + i)
            for i in range(n_estimators)
        ]

    def fit(self, X, y):
        for model in self.models:
            # Sample with replacement for each tree (bootstrap sampling)
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            model.fit(X_sample, y_sample)

    def predict(self, X):
        # Average predictions from all trees
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

    def call(self, inputs, training=False):
        return self.predict(inputs)



class ElasticNetModel(tf.keras.Model):
    def __init__(self, input_dim, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetModel, self).__init__()
        self.dense = tf.keras.layers.Dense(1)  # Single neuron for regression output
        self.alpha = alpha  # Regularization strength
        self.l1_ratio = l1_ratio  # Balance between L1 and L2 regularization

    def call(self, inputs, training=False):
        return self.dense(inputs)

    def compute_loss(self, y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        l1_loss = tf.reduce_sum(tf.abs(self.dense.kernel))  # L1 regularization
        l2_loss = tf.reduce_sum(tf.square(self.dense.kernel))  # L2 regularization
        elastic_net_loss = mse_loss + self.alpha * (self.l1_ratio * l1_loss + (1 - self.l1_ratio) * l2_loss)
        return elastic_net_loss


# class ResNet50Model(tf.keras.Model):
#     def __init__(self):
#         super(ResNet50Model, self).__init__()
        
#         self.resnet = tf.keras.applications.ResNet50(
#             include_top=False,
#             weights="imagenet",
#             input_shape=(3017),
#             pooling="avg"
#         )
#         self.dense1 = tf.keras.layers.Dense(units=128, activation="relu")
#         self.dense2 = tf.keras.layers.Dense(units=64, activation="relu")
#         self.final_dense = tf.keras.layers.Dense(units=1)

#     def call(self, inputs, training=False):
#         x = self.resnet(inputs, training=training)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         output = self.final_dense(x)
#         return output
