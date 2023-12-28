import numpy as np
import tensorflow as tf

test_number = 1

model = tf.keras.models.load_model(f'Results/Attack/rte_case14_realistic/TG/test_{test_number}/my_model')

x_test = np.random.rand(24,20) # input window size by power line number
x_test= np.expand_dims(x_test, axis = 0)
x_test= np.expand_dims(x_test, axis = 3)
y_pred = model.predict(x_test)
y_pred = y_pred[0,:,:]