
import tensorflow as tf

class ResNet50Model(tf.keras.Model):
    def __init__(self):
        super(ResNet50Model, self).__init__()
        
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            # weights="imagenet",
            weights=None,
            input_shape=(224, 224, 3),
            pooling="avg"
        )
        self.dense1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=64, activation="relu")
        self.final_dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=False):
        x = self.resnet(inputs, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.final_dense(x)
        return output


class ResNet50ClassificationModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet50ClassificationModel, self).__init__()
        
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            # weights="imagenet",
            weights=None,
            input_shape=(224, 224, 3),
            pooling="avg"
        )
        self.dense1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=64, activation="relu")
        # Updated final dense layer for classification
        self.final_dense = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.resnet(inputs, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.final_dense(x)
        return output