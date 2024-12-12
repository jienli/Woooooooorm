import tensorflow as tf

class SepConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="same", **kwargs):
        super(SepConv2D, self).__init__(**kwargs)
        self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.pointwise = tf.keras.layers.Conv2D(filters, kernel_size=1, padding="same")

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class ConvEmbed(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(ConvEmbed, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        return x  
    
class AttentionMatrix(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionMatrix, self).__init__()

    def call(self, inputs):
        Q, K = inputs
        embedding_size_keys = tf.shape(K)[-1]

        score = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(embedding_size_keys, tf.float32))
        atten_weight = tf.nn.softmax(score, axis=-1)
        return atten_weight

class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, projection_dim, kernel_size, strides=[1, 2, 2], **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.query_conv = SepConv2D(
            filters=projection_dim, kernel_size=kernel_size, strides=strides[0], padding="same"
        )
        self.key_conv = SepConv2D(
            filters=projection_dim, kernel_size=kernel_size, strides=strides[1], padding="same"
        )
        self.value_conv = SepConv2D(
            filters=projection_dim, kernel_size=kernel_size, strides=strides[2], padding="same"
        )
        self.atten_matrix = AttentionMatrix()
        
    def call(self, query, key, value):
        queries = self.query_conv(query)
        keys = self.key_conv(key)
        values = self.value_conv(value)

        batch_size = tf.shape(queries)[0]
        height = queries.shape[1]
        width = queries.shape[2]
        channels = queries.shape[3]

        queries = tf.reshape(queries, [batch_size, -1, channels]) 
        keys = tf.reshape(keys, [batch_size, -1, channels])
        values = tf.reshape(values, [batch_size, -1, channels])

        atten_weight = self.atten_matrix([queries, keys])
        output = tf.matmul(atten_weight, values)

        output = tf.reshape(output, [batch_size, height, width, channels])
        return output

class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, kernel_size=3, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        projection_dim = embed_dim // num_heads

        self.attention_heads = [AttentionHead(projection_dim, kernel_size) for _ in range(num_heads)]
        self.linear = tf.keras.layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding="same")

    def call(self, query, key, value):
        head_outputs = [head(query, key, value) for head in self.attention_heads]
        output = tf.concat(head_outputs, axis=-1)
        output = self.linear(output)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadedAttention(embed_dim, num_heads)

        self.feed_forward = tf.keras.Sequential([
                tf.keras.layers.Dense(embed_dim * 2, activation=tf.nn.gelu),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(embed_dim),
                tf.keras.layers.Dropout(dropout)
            ])
        
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        # prenorm
        normed_inputs = self.norm1(inputs)
        attention_output = self.attention(normed_inputs, normed_inputs, normed_inputs)
        attention_residual = inputs + attention_output

        normed_attention_residual = self.norm2(attention_residual)
        feed_forward_output = self.feed_forward(normed_attention_residual)
        output = attention_residual + feed_forward_output

        return output
    
class CvTBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, embed_dim, num_heads, depth=1):
        super(CvTBlock, self).__init__()

        self.conv = ConvEmbed(filters, kernel_size, strides)
        self.transformer_layers = [
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ]

    def call(self, inputs):
        x = self.conv(inputs)
        for transformer in self.transformer_layers:
            x = transformer(x)
        return x
    
class CvTModel(tf.keras.Model):
    def __init__(self,
                 num_classes=1,
                 num_blocks=3,
                 embed_dims=[32, 64, 128],
                 kernel_size = [7, 3, 3],
                 num_heads=[1, 3, 6],
                 strides=[2, 2, 2],
                 depths=[1, 2, 4],
                 **kwargs):
        
        super(CvTModel, self).__init__(**kwargs)

        self.flip = tf.keras.layers.RandomFlip()

        self.CvT_blocks = []
        for i in range(num_blocks):
            self.CvT_blocks.append(
                CvTBlock(
                    filters=embed_dims[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    depth=depths[i]
                )
            )

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        if training:
            inputs = self.flip(inputs)

        x = inputs
        for block in self.CvT_blocks:
            x = block(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
