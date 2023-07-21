# -*- coding: utf-8 -*-
"""
Image Generation using GAN with FER 2013 Dataset
# Load  and Preprocess FER 2013 dataset
# Define Generator and Discriminator model
# Define loss functions and optimizers
# Train the GAN model
# Calculate FID score
"""
   
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Load and preprocess the FER 2013 Dataset
# Path to the FER 2013 CSV file
csv_path = 'FER 2013 Dataset.csv'

# Load the CSV file
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=str)

# Extract image pixels and labels from the CSV data
images = np.array([np.fromstring(image, dtype=int, sep=' ') for image in data[:, 1]])
labels = np.array(data[:, 0], dtype=int)

# Normalize images to range [-1, 1]
images = (images.astype('float32') - 127.5) / 127.5
images = np.reshape(images, (-1, 48, 48, 1))

# Generator Model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(6 * 6 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((6, 6, 256)))
    assert model.output_shape == (None, 6, 6, 256)  # None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 12, 12, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 24, 24, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 48, 48, 1)

    return model

# Discriminator Model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[48, 48, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Initialize generator and discriminator
generator = make_generator_model()
discriminator = make_discriminator_model()

# Define batch size and number of epochs
BATCH_SIZE = 128
EPOCHS = 50

# Define training loop
num_examples = images.shape[0]
num_batches = num_examples // BATCH_SIZE

# Lists to store losses for plotting
gen_loss_history = []
disc_loss_history = []

# Show original images
fig, axs = plt.subplots(4, 4)
count = 0
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(images[count, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        count += 1
plt.suptitle('Original Images')
plt.show()

# Compute feature embeddings for real images
real_embeddings = np.mean(images, axis=(1, 2, 3))

def calculate_fid(real_embeddings, generated_embeddings):
    # Calculate mean and covariance for real and generated embeddings
    mu_real = np.mean(real_embeddings, axis=0)
    real_embeddings_2d = np.reshape(real_embeddings, (real_embeddings.shape[0], -1))
    sigma_real = np.cov(real_embeddings_2d, rowvar=False) + np.eye(real_embeddings_2d.shape[1]) * 1e-6
    
    mu_generated = np.mean(generated_embeddings, axis=0)
    generated_embeddings_2d = np.reshape(generated_embeddings, (generated_embeddings.shape[0], -1))
    sigma_generated = np.cov(generated_embeddings_2d, rowvar=False) + np.eye(generated_embeddings_2d.shape[1]) * 1e-6
    
    # Calculate squared Frobenius norm between mean and covariance
    diff = mu_real - mu_generated
    cov_sqrt_real = linalg.sqrtm(sigma_real)
    cov_sqrt_generated = linalg.sqrtm(sigma_generated)
    cov_sqrt_product = cov_sqrt_real.dot(cov_sqrt_generated)

    # Calculate FID score
    fid_score = np.sum(diff ** 2) + np.trace(sigma_real + sigma_generated - 2 * cov_sqrt_product)

    return fid_score


# Training loop
for epoch in range(EPOCHS):
    gen_loss_sum = 0
    disc_loss_sum = 0
    generated_embeddings = []
    for batch_idx in range(num_batches):
        batch_images = images[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        gen_loss, disc_loss = train_step(batch_images)
        gen_loss_sum += gen_loss
        disc_loss_sum += disc_loss
        
        # Compute feature embeddings for generated images
        generated_images = generator(tf.random.normal([BATCH_SIZE, 100]), training=False)
        generated_embeddings.append(np.mean(generated_images, axis=(1, 2, 3)))

    # Compute average losses for the epoch
    gen_loss_avg = gen_loss_sum / num_batches
    disc_loss_avg = disc_loss_sum / num_batches

    # Append losses to history
    gen_loss_history.append(gen_loss_avg)
    disc_loss_history.append(disc_loss_avg)

    # Generate and save synthetic images every few epochs
    if epoch % 10 == 0:
        noise = tf.random.normal([16, 100])
        generated_images = generator(noise, training=False)

        fig, axs = plt.subplots(4, 4)
        count = 0
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.suptitle('Generated Images (Epoch {})'.format(epoch))
        plt.show()
        
        # Calculate FID
        generated_embeddings = np.concatenate(generated_embeddings, axis=0)
        fid = calculate_fid(real_embeddings, generated_embeddings)
        print("FID at epoch {}: {}".format(epoch, fid))
    
    # Print epoch and losses
    print("Epoch:", epoch)
    print("Generator Loss:", gen_loss_avg)
    print("Discriminator Loss:", disc_loss_avg)
    
# Plot loss curve
plt.plot(range(EPOCHS), gen_loss_history, label='Generator Loss')
plt.plot(range(EPOCHS), disc_loss_history, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()