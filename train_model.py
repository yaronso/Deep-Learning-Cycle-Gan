import time
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, LeakyReLU)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

# Determine image resolution.
IMAGE_RES = [256, 256]


# Loading data from Google Drive (originally imported from Kaggle, in tfrec format).
def loading_data_from_drive():
    
    # When running in Kaggle notebook
    # The dataset has to be loaded from GCS if you want to use it on TPU.
    '''GCS_PATH = KaggleDatasets().get_gcs_path()
    monet_paintings = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
    photos = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))'''
    
    # When Using Google Drive
    monet_paintings = tf.io.gfile.glob(str('/content/drive/MyDrive/Monet_CycleGAN_DL/monet_tfrec/*.tfrec'))
    photos = tf.io.gfile.glob(str('/content/drive/MyDrive/Monet_CycleGAN_DL/photo_tfrec/*.tfrec'))
    
    return monet_paintings, photos


# Converting tfrec image to jpeg image.
def tfrec_to_jpeg(image):
    # Conversion and scaling is being made on the images.
    return tf.reshape((tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32) / 127.5) - 1, [*IMAGE_RES, 3])


# This function is responsible of each tfrec in out dataset to jpeg format.
def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    return tfrec_to_jpeg(tf.io.parse_single_example(example, tfrecord_format)['image'])


# Sampling 30 images algorithm
def get_sampled_images(entire_data):
    dataset_length = len(list(entire_data))
    print("The input dataset length is: ", dataset_length)
    new_lst = []
    rand_indexes = []
    i = 0
    while i < 300:
        rand_i = random.randint(0, 299)
        if rand_i not in rand_indexes:
            i += 1
            rand_indexes.append(rand_i)

    for i in range(30):
        new_lst.append(list(entire_data)[rand_indexes[i]])
    new_ds = tf.data.Dataset.from_tensor_slices(new_lst)
    data_stock = tf.data.Dataset.zip(new_ds.take(30))
    print('final data stock length: ', len(list(data_stock)))
    #data_stock = entire_data.take(30)
    #print('Test data stock length: ', len(list(data_stock)))
    return data_stock


# Loading monet painting and photos.
def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset

# Load the photos for the prediction phase.
def load_dataset_predict(filenames):
    dataset = tf.data.TFRecordDataset(filenames[:8])
    dataset = dataset.map(read_tfrecord)
    return dataset


# A 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
def down_sample_discriminator(val, k, apply_normalization=True):
    layer = Conv2D(k, (4, 4), strides=2, padding='same', kernel_initializer=weight_initializer)(val)
    if apply_normalization:
        layer = InstanceNormalization(axis=-1)(layer)
    return LeakyReLU(0.2)(layer)


# Setting discriminator, with the consent architecture, including down sampling.
def discriminator():
    # input: discriminator receives an image, and should decide if it is fake or not.
    input_val = Input(shape=(img_rows, img_cols, channels))
    # Down sampling process
    # In the first layer, disable instance normalization
    disc_model = down_sample_discriminator(input_val, 64, False)
    disc_model = down_sample_discriminator(disc_model, 128)
    disc_model = down_sample_discriminator(disc_model, 256)
    disc_model = down_sample_discriminator(disc_model, 512)
    # The last layer of the discriminator.
    disc = Conv2D(1, (4, 4), padding='same', kernel_initializer=weight_initializer)(disc_model)
    return Model(input_val, disc)


# A 3x3 convolution network, with variable K filters with stride 2, and Relu activation function.
def down_sample_generator(k, apply_normalization=True):
    layer = Sequential()
    layer.add(Conv2D(k, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer))
    # apply instance normalization to the layers
    if apply_normalization:
        layer.add(InstanceNormalization(axis=-1))
    # Adding the activation function.
    layer.add(Activation('relu'))
    return layer


# A 3x3 convolution network, with K filters with stride 2, and Relu activation function.
def up_sample_generator(k):
    layer = Sequential()
    layer.add(Conv2DTranspose(k, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer))
    layer.add(InstanceNormalization(axis=-1))
    layer.add(Activation('relu'))
    return layer


# Our Generator model based on U-Net architecture 
def generator():
    generator_input = Input(shape=(img_rows, img_cols, channels))
    # Setting the layers for the encoding part of the model.
    encoder_layers = [
        down_sample_generator(64, False),
        down_sample_generator(128),
        down_sample_generator(256),
        down_sample_generator(512),
        down_sample_generator(512),
        down_sample_generator(512),
        down_sample_generator(512),
        down_sample_generator(512)
    ]
    # Setting the layers for the decoding part of the model.
    decoder_layers = [
        up_sample_generator(512),
        up_sample_generator(512),
        up_sample_generator(512),
        up_sample_generator(512),
        up_sample_generator(256),
        up_sample_generator(128),
        up_sample_generator(64)
    ]
    res_generator = generator_input
    skips = []
    
    # Adding all of the encoder layers, and keep track of them for skip connections.
    for layer in encoder_layers:
        res_generator = layer(res_generator)
        skips.append(res_generator)
        
    # Reverse the list for looping, and to get rid of the layer that directly connects to the decoder.
    skips = skips[::-1][1:]
    
    # Add all the decoder layers with the skip connections of each layer.
    for skip_layer, layer in zip(skips, decoder_layers):
        res_generator = layer(res_generator)
        res_generator = Concatenate()([res_generator, skip_layer])
    
    # Building the last layer.
    res_generator = Conv2DTranspose(channels, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer,
                                    activation='tanh')(res_generator)
    # Finally, compose the model.
    return Model(generator_input, res_generator)


# The discriminator loss function below compares real images to a matrix of 1s and fake images to a matrix of 0s.
def discriminator_loss(real, generated):
    # Multiplied by 0.5 so that it will train at half-speed (better performance).
    return (loss(tf.ones_like(real), real) + loss(tf.zeros_like(generated), generated)) * 0.5


# Measures how real the discriminator believes the fake image is
def generator_loss(validity):
    return loss(tf.ones_like(validity), validity)


# Measures similarity of two images, Used for cycle and identity loss.
def image_similarity(real_image, cycled_image):
    return tf.reduce_mean(tf.abs(real_image - cycled_image))


@tf.function
def train_model(real_x, real_y):
    # we will be using a tf.GradientTape object to compute the losses multiple times.
    with tf.GradientTape(persistent=True) as tape:
        # Setup the Discriminator Y loss:
        # G(X) = fake_y
        fake_y = generator_g(real_x, training=True)
        # The Discriminator receive a fake generated image: D_Y(fake_y) = fake_y_validity (the prediction on fake_y)
        fake_y_validity = discriminator_y(fake_y, training=True)
        # Calculate the Discriminator based on its prediction on the fake_y image, and the real_y image.
        # Cal D_Y_LOSS accordingly (Real_Y, fake_y_validity)
        dis_y_loss = discriminator_loss(discriminator_y(real_y, training=True), fake_y_validity)
        
        # Update Discriminator Y Gradients
        with tape.stop_recording():
            # Get grads.
            discriminator_y_gradients = tape.gradient(dis_y_loss, discriminator_y.trainable_variables)
            # Update procedure.
            dis_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))
            
        # Setup Discriminator X loss:
        # F(Y=photo) = (FAKE_X=monet style photo). 
        # Generator F will be in used for the prediction stage.
        fake_x = generator_f(real_y, training=True)
        # D_Y(FAKE_X) = FAKE_X_VALIDITY
        fake_x_validity = discriminator_x(fake_x, training=True)
        # Cal D_X_LOSS accordingly (Real_X, FAKE_X_VALIDITY)
        dis_x_loss = discriminator_loss(discriminator_x(real_x, training=True), fake_x_validity)
        
        # Update Discriminator X Gradients
        with tape.stop_recording():
            # Get grads.
            discriminator_x_gradients = tape.gradient(dis_x_loss, discriminator_x.trainable_variables)
            # Update procedure.
            dis_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
            
        # Setup the Generators losses:
        # Evaluating how real the discriminator believes the generated images are.
        gen_g_adv_loss = generator_loss(fake_y_validity)
        gen_f_adv_loss = generator_loss(fake_x_validity)
        
        # Setup cycle losses: attempt to translate fake_y and fake_x to real_y and real_x.
        # Should be: G(F(y)) = y, and F(G(x)) = x.
        cycle_x = generator_f(fake_y, training=True)
        cycle_x_loss = image_similarity(real_x, cycle_x)
        cycle_y = generator_g(fake_x, training=True)
        cycle_y_loss = image_similarity(real_y, cycle_y)
        
        # Setup identity loss of x:
        id_x = generator_f(real_x, training=True)
        id_x_loss = image_similarity(real_x, id_x)
        # Setup identity loss of y.
        
        id_y = generator_g(real_y, training=True)
        id_y_loss = image_similarity(real_y, id_y)
        
        # Sum up all the Generators losses:
        gen_g_loss = gen_g_adv_loss + (cycle_x_loss + cycle_y_loss) * lambda_weight + id_y_loss * 0.5 * lambda_weight
        gen_f_loss = gen_f_adv_loss + (cycle_x_loss + cycle_y_loss) * lambda_weight + id_x_loss * 0.5 * lambda_weight
        
        # Update the Gradients of the Generators:
        with tape.stop_recording():
            # Gradient of first generator - G gen.
            generator_g_gradients = tape.gradient(gen_g_loss, generator_g.trainable_variables)
            gen_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
            
            # Gradient of second generator - F gen.
            generator_f_gradients = tape.gradient(gen_f_loss, generator_f.trainable_variables)
            gen_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
            
        return dis_y_loss, dis_x_loss, gen_g_loss, gen_f_loss


# Prediction generator function.
def create_predictions(predict_y):
    _, frame = plt.subplots(5, 2, figsize=(12, 12))
    # Total of 5 predictions.
    for i, img in enumerate(predict_y.take(5)):
        prediction = generator_f(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
        frame[i, 0].imshow(img)
        frame[i, 1].imshow(prediction)
        frame[i, 0].set_title("Input photo " + str(i + 1))
        frame[i, 1].set_title("Monet version " + str(i + 1))
        frame[i, 0].axis("off")
        frame[i, 1].axis("off")
    plt.show()

# The following function plot the Generator F loss in each epoch.
def plot_loss(loss_lst):
  plt.plot(loss_lst)
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.suptitle('Monet Generator (F) Loss')
  plt.show()


# The full training process of our GAN models
def training_process(epochs):
    # The main loop of the training process
    gen_f_loss_lst = []
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        start = time.time()

        # Run on each batch
        for batch, (x, y) in enumerate(tf.data.Dataset.zip((monet_ds, photo_ds))):
            if batch % 10 == 0:
                print("current batch: ", batch)
            # Training step
            dis_y_loss, dis_x_loss, gen_g_loss, gen_f_loss = train_model(
                tf.reshape(x, (1, img_rows, img_cols, channels)),
                tf.reshape(y, (1, img_rows, img_cols, channels)))
        print(f'Loss in epoch {epoch}: dis_y_loss: {dis_y_loss}, dis_x_loss: {dis_x_loss}, gen_f_loss: {gen_f_loss}, gen_g_loss: {gen_g_loss}')
        gen_f_loss_lst.append(gen_f_loss)
        print("Epoch duration: %.2f" % ((time.time() - start) / 60))
    
    # Invoke the function that plot the loss of Generator F for analysis.
    plot_loss(gen_f_loss_lst)


if __name__ == '__main__':
    # Loading out data from source, and tuning it to be worked as a tf dataset.
    monets, photos = loading_data_from_drive()
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    # Setting our datasets and pred
    monet_ds = load_dataset(monets).batch(1)
    monet_ds = get_sampled_images(monet_ds)
    photo_ds = load_dataset(photos).batch(1)
    predict_y = load_dataset_predict(photos).batch(1)

    # Setting number of epochs for the training process.
    epochs = 50
    # Setting the relative importance of the cycle loss to the adversarial loss
    lambda_weight = 5
    # Setting dimensions of target image.
    img_rows, img_cols, channels = 256, 256, 3
    # Setting an initializer for the model weights
    weight_initializer = RandomNormal(stddev=0.02)

    # The first generator G will map monet photo to regular photo
    # The second Generator F will map original image to monet style photo
    gen_g_optimizer = gen_f_optimizer = Adam(learning_rate=0.0002, beta_1=0.5) 
    dis_x_optimizer = dis_y_optimizer = Adam(learning_rate=0.0006, beta_1=0.5)
    
    # Adam(learning_rate=0.00006, beta_1=0.5)
    # tf.keras.optimizers.Adagrad(learning_rate=0.0002, initial_accumulator_value=0.5, epsilon=1e-07, name="Adagrad")
    # tf.keras.optimizers.Adadelta(learning_rate=0.00006, rho=0.5, epsilon=1e-07, name='Adadelta') 
    
    
    # Initialize the models.
    # Generators init.
    generator_g = generator()
    generator_f = generator()
    
    # Discriminators init.
    discriminator_x = discriminator()
    discriminator_y = discriminator()

    # Set the loss function, will be used in both discriminator and generator loss functions.
    loss = BinaryCrossentropy(from_logits=True)
    
    # Training the model, with predefined number of epochs.
    training_process(epochs)

    generator_f.save('/content/drive/MyDrive/Monet_CycleGAN_DL')
    print("The model was saved in Google drive's data directory")
    
    # After training the model, call generate images method, to check our results by using Generator F trained model.
    create_predictions(predict_y)

