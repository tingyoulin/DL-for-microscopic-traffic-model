# Importing dependencies numpy and keras
import numpy
import time
from module import Environment, Vehicle, Simulation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

start_load = time.time()
# 生成時間序列資料 load traffic data
sim = Simulation()
X, y, env_list = sim.data_generator(base_dir="C:/Users/User/Google Drive/Master Thesis/Data", input_var=None)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 一些參數
BATCH_START = 0
TIME_STEPS = X_train.shape[1]
BATCH_SIZE = 20
INPUT_SIZE = X_train.shape[2]
OUTPUT_SIZE = y_train.shape[2]
CELL_SIZE = 20      # hidden layer unit數
LR = 6E-3

start_model = time.time()
# defining the LSTM model
model = Sequential()
model.add(
    LSTM(
        300,
        input_shape=(TIME_STEPS, INPUT_SIZE),
        return_sequences=True
    )
)
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(
    Dense(
        OUTPUT_SIZE,
        activation=None
    )
)


# Loss function: x, y座標的 loss
def loss_function(y_true, y_pred):
    return 1


model.compile(loss=loss_function, optimizer=Adam(LR))

start_fit = time.time()
"""
# fitting the model
model.fit(X_modified, Y_modified, epochs=1, batch_size=30)

start_randseed = time.time()

# picking a random seed
start_index = numpy.random.randint(0, len(X)-1)
new_string = X[start_index]

start_gen = time.time()

# generating characters
for i in range(50):
    x = numpy.reshape(new_string, (1, len(new_string), 1))
    x = x / float(len(unique_chars))

    # predicting
    pred_index = numpy.argmax(model.predict(x, verbose=0))
    char_out = int_to_char[pred_index]
    seq_in = [int_to_char[value] for value in new_string]
    print(char_out)

    new_string.append(pred_index)
    new_string = new_string[1:len(new_string)]
"""
print(
    "load text:", start_model - start_load,
    # "\nmapping characters with integers:", start_data - start_map,
    # "\npreparing input and output dataset:", start_reshape - start_data,
    # "\nreshaping, normalizing and one hot encoding:", start_model - start_reshape,
    "\ndefining the LSTM model:", start_fit - start_model,
    # "\nfitting the model:", start_randseed - start_fit,
    # "\npicking a random seed:", start_gen - start_randseed,
    # "\ngenerating characters:", time.time() - start_gen
)
