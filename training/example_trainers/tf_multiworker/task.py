"""TensorFlow Multi-worker strategy training script."""
# adapted from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
import os
import json

import tensorflow as tf

import util

PER_WORKER_BATCH_SIZE = 64


def main():
  tf_config_str = os.environ['TF_CONFIG']
  print(type(tf_config_str))
  print(tf_config_str)
  tf_config = json.loads(tf_config_str)
  num_workers = len(tf_config['cluster']['worker'])

  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  global_batch_size = PER_WORKER_BATCH_SIZE * num_workers
  multi_worker_dataset = util.mnist_dataset(global_batch_size)

  with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = util.build_and_compile_cnn_model()

  multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)

  # print accuracy
  _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  _, accuracy = multi_worker_model.evaluate(x_test, y_test, )
  print('Accuracy: %f' % accuracy)


if __name__ == "__main__":
  main()
