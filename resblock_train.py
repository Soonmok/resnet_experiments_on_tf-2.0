import tensorflow as tf

from models import get_model
from utils import preprocess

# set tensorflow dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64).map(
    preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(64).map(
    preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)

# load models
resnet_model = get_model(num_blocks=10, resnet=True)
baseline_model = get_model(num_blocks=10, resnet=False)

# define losses and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(x_train, y_train):
    with tf.GradientTape() as resnet_tape, tf.GradientTape() as baseline_tape:
        resnet_y_pred = resnet_model(x_train)
        resnet_loss = loss_object(y_train, resnet_y_pred)
        baseline_y_pred = baseline_model(x_train)
        baseline_loss = loss_object(y_train, baseline_y_pred)

    resnet_gradient_of_resnet = resnet_tape.gradient(resnet_loss, resnet_model.trainable_variables)
    optimizer.apply_gradients(zip(resnet_gradient_of_resnet, resnet_model.trainable_variables))

    baseline_gradient_of_resnet = baseline_tape.gradient(baseline_loss, baseline_model.trainable_variables)
    optimizer.apply_gradients(zip(baseline_gradient_of_resnet, baseline_model.trainable_variables))

    template = 'resnet {}  /  baseline {}'
    tf.print("resnet -> ", tf.reduce_mean(tf.abs(resnet_gradient_of_resnet[10])), "baseline -> ", tf.reduce_mean(tf.abs(baseline_gradient_of_resnet[10])))


for idx, (x_train, y_train) in enumerate(train_dataset):
    train_step(x_train, y_train)