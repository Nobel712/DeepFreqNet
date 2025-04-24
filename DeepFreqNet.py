
image_size = 224  # Example size, can be adjusted

# Define the input layer
inputs = Input(shape=(image_size, image_size, 3))

# Block 1
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2 with Inception-like structure
x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x3 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
x = layers.Concatenate(axis=-1, name='block2_concat')([x1, x2, x3])
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3 with Depthwise Separable Convolution
x = layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4 with Residual Connection
r = layers.Conv2D(256, (1, 1), padding='same')(x)
x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([r, x])  # Residual connection
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Fully Connected Layers
x = layers.Flatten(name='flatten')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(30, activation='softmax', name='predictions')(x)

# Create the model
model = models.Model(inputs=inputs, outputs=outputs)
