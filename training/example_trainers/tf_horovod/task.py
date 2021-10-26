"""TensorFlow Horovod training script."""
# adapted from https://horovod.readthedocs.io/en/stable/tensorflow.html
import numpy as np

import horovod.tensorflow as hvd
import tensorflow as tf

PER_WORKER_BATCH_SIZE = 64


def mnist_dataset(batch_size):
  """Prepare dataset."""
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset


def build_and_compile_cnn_model():
  """Build and compile model."""
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


@tf.function
def training_step(images_batch, labels_batch, first_batch):
  """Training step."""
  with tf.GradientTape() as tape:
    probs = mnist_model(images_batch, training=True)
    loss = loss_function(labels_batch, probs)

  # Horovod: add Horovod Distributed GradientTape.
  tape = hvd.DistributedGradientTape(tape)

  grads = tape.gradient(loss, mnist_model.trainable_variables)
  opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

  # Horovod: broadcast initial variable states from rank 0 to all other
  # processes.
  # This is necessary to ensure consistent initialization of all workers when
  # training is started with random weights or restored from a checkpoint.
  #
  # Note: broadcast should be done after the first gradient step to ensure
  # optimizer initialization.
  if first_batch:
    hvd.broadcast_variables(mnist_model.variables, root_rank=0)
    hvd.broadcast_variables(opt.variables(), root_rank=0)

  return loss


# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Build model and dataset
dataset = mnist_dataset(PER_WORKER_BATCH_SIZE)
mnist_model = build_and_compile_cnn_model()
loss_function = tf.losses.SparseCategoricalCrossentropy()
opt = tf.optimizers.Adam(0.001 * hvd.size())

checkpoint_dir = './checkpoints'
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


# Horovod: adjust number of steps based on number of GPUs.
for batch, (images, labels) in enumerate(dataset.take(10000 // hvd.size())):
  loss_value = training_step(images, labels, batch == 0)

  if batch % 10 == 0 and hvd.local_rank() == 0:
    print('Step #%d\tLoss: %.6f' % (batch, loss_value))

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if hvd.rank() == 0:
  checkpoint.save(checkpoint_dir)
