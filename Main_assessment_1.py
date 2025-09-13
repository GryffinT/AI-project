from Main_classification import statement
from Main_classification import classifications
from tensorFlow import Keras
from tensorFlow.keras import layers

# Network with 1 layer
model = keras.Sequential([
  layers.dense(units=2, input_shape=[1]) # units = outputs, input_shape = num of features/dimension of input, so 1 input.
])
