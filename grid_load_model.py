import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


trained_model = tf.keras.models.load_model('Attack_dataset/rte_case14_realistic/PLS/test_1/my_model')

