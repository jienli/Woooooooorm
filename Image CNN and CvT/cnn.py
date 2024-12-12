import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ConvBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-3)
        self.activation = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=0.2)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

# Define the architecture of the CNN model
class CNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.flip = tf.keras.layers.RandomFlip()
        
        self.convb1 = ConvBlock(64, 3)
        self.convb2 = ConvBlock(128, 3)
        self.convb3 = ConvBlock(256, 3)

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)
        
        self.final_dense = tf.keras.layers.Dense(units=num_classes)

    def call(self, inputs, training=False):
        if training:
            inputs = self.flip(inputs)

        x = self.convb1(inputs, training=training)
        x = self.convb2(x, training=training)
        x = self.convb3(x, training=training)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        output = self.final_dense(x)

        return output