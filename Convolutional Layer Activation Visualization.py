# -*- coding: utf-8 -*-
"""
This code defines the Convolutional Layer Activation Visualization 
# Load and preprocess the image from FER 2013 dataset
# Load the model with architecture and weights
# Create a list to store the intermediate outputs
# Predict the image and get the intermediate activations
# Visualize the intermediate activations

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load and preprocess the image from FER 2013 dataset
image_path = 'FER 2013 Source_Image.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
image = cv2.resize(image, (48, 48))  # Resize the image to (48, 48)
image = np.expand_dims(image, axis=-1)  # Add a channel dimension
image = image.astype('float32') / 255.0  # Convert to float32 and normalize

# Load the model with architecture and weights
model = tf.keras.models.load_model('savemodelandweights.hdf5')

# Create a list to store the intermediate outputs
layer_outputs = [layer.output for layer in model.layers]

# Create a new model that outputs the intermediate activations
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Predict the image and get the intermediate activations
activations = activation_model.predict(image[np.newaxis, ...])

# Visualize the intermediate activations
for layer_activation in activations:
    if len(layer_activation.shape) == 4:  # Convolutional layer activations
        num_activations = layer_activation.shape[-1]
        square_dim = int(np.ceil(np.sqrt(num_activations)))
        fig, axes = plt.subplots(square_dim, square_dim)
        for i, ax in enumerate(axes.flat):
            if i < num_activations:
                ax.imshow(layer_activation[0, :, :, i], cmap='gray')
            ax.axis('off')
        plt.show()



