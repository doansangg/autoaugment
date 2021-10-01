import data_custom
import data_utils
import tensorflow as tf
hparams = tf.contrib.training.HParams(
      train_size=50000,
      validation_size=0,
      eval_test=1,
      dataset="cifar10",
      data_path='cifar-10-binary/cifar-10-batches-py',
      batch_size=2,
      gradient_clipping_by_global_norm=5.0)
data = data_utils.DataSet(hparams)
# for i in range(10):
#       train_images, train_labels = data.next_batch()
      #print(train_images)
# for i,lod in data:
#       print(lod.shape)
#print(data)