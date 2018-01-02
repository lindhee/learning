# -*- coding: UTF-8 -*-
# Load a batch of cat/dog images into TensorFlow, and add labels derived from the file names
# Magnus Lindhé, 2018
# The code for loading images is based on 
# "https://gist.github.com/eerwitt/518b0c9564e500b4b50f"

import tensorflow as tf

# imageFileNamePattern is a string with a regexp that matches 
# all file names that should be loaded.
# Returns:
# (image_batch, label_batch)  
def loadImages(imageFileNamePattern):

    print("Reading files: " + imageFileNamePattern)

    # Make a queue of file names
    # We could add shuffling and a limit on no of epochs
    name_list = tf.train.match_filenames_once(imageFileNamePattern)
    filename_queue = tf.train.string_input_producer(name_list) 

    # Each image is fetched by reading a whole file
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue
    file_name, image_file = image_reader.read(filename_queue)

    # Extract the image label from the file name
    # Cat = 0, Dog = 1
    sub_strings = tf.string_split([file_name],"/")
    partial_file_name = sub_strings.values[-1] # File name, with path stripped off
    sub_strings2 = tf.string_split([partial_file_name],".")
    animal_type = sub_strings2.values[0] # First part of name "cat.X.jpg"
    label = tf.where(tf.equal(animal_type,"dog"),1,0)
    float_label = tf.reshape(tf.cast(label,dtype=tf.float32),[1])

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    # NOTE: We make it a 2D tensor by decoding it into grayscale.
    image = tf.image.decode_jpeg(image_file,channels=1)

    # Pre-processing
    float_image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_images(float_image, [200, 200])
    resized_image.set_shape([200, 200, 1])

    # Subtract off the mean and divide by the variance of the pixels.
    normalized_image = tf.image.per_image_standardization(resized_image)
    squeezed_image = tf.reshape(normalized_image,[200, 200])

    # Print the image and label types
    print("Image: " + str(squeezed_image.shape) + " of type " + str(squeezed_image.dtype))
    print("Label: " + str(float_label.shape) + " of type " + str(float_label.dtype))

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10
    batch_size = 5
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
      [squeezed_image, float_label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
      
    return (image_batch, label_batch)  
      
