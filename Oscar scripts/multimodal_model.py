import tensorflow as tf


class MultimodalModel(tf.keras.Model):
    def __init__(self, resnet_models, mlp_model):
        super(MultimodalModel, self).__init__()
        self.resnet_models = resnet_models  # List of ResNet models
        self.mlp = mlp_model
        self.concat = tf.keras.layers.Concatenate()
        self.final_dense1 = tf.keras.layers.Dense(units=64, activation="relu")
        self.final_dense2 = tf.keras.layers.Dense(units=32, activation="relu")
        self.output_layer = tf.keras.layers.Dense(units=1)  # Regression output

    def call(self, inputs, training=False):
        # Split inputs
        image_inputs_list, tabular_inputs = inputs

        # Process each image through its respective ResNet model
        resnet_outputs = [resnet(image_input, training=training) for resnet, image_input in zip(self.resnet_models, image_inputs_list)]

        # Process tabular data through MLP
        mlp_output = self.mlp(tabular_inputs, training=training)
        
        # Ensure all outputs have the same shape
        resnet_outputs = [tf.keras.layers.Flatten()(output) for output in resnet_outputs]  # Flatten ResNet outputs
        mlp_output = tf.keras.layers.Flatten()(mlp_output)  # Flatten MLP output

        # Concatenate outputs
        combined = self.concat(resnet_outputs + [mlp_output])

        # Process through final dense layers
        x = self.final_dense1(combined)
        x = self.final_dense2(x)
        output = self.output_layer(x)
        return output