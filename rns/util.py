import tensorflow as tf

def merge_summaries(sd, id):
    summaries = []
    for key in sd.keys():
        summaries.append(tf.summary.scalar(key, sd[key]))
    for key in id.keys():
        summaries.append(tf.summary.image(key, id[key]))
    return tf.summary.merge(summaries)

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images

def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=3)

def cartesian_product(a,b):
    a, b = a[None, :, None], b[:, None, None]
    prod = tf.concat([b + tf.zeros_like(a), tf.zeros_like(b) + a], axis = 2)
    #new_shape = tf.stack([-1, tf.shape(cartesian_product)[-1]])
    #cartesian_product = tf.reshape(cartesian_product, new_shape)
    prod = tf.reshape(prod, [-1])
    return prod
