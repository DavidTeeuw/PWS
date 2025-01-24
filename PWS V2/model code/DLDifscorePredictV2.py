import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras import Model # type: ignore
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import numpy as np # type: ignore

# Data ophalen en klaarmaken om het model te trainen
df = pd.read_csv(r'datasets/dataset_frames_vector_difscore.csv', header=None, names=['vectors', 'difscore'])
df['vectors'] = df['vectors'].apply(lambda x: np.array(list(map(float, x.split()))))

X = np.array(df['vectors'].tolist())
y = df['difscore'] - 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architectuur
def difscorePredictModel():
    # Model 1
    inputLayer1 = Input(shape= (1330, ))
    denseLayer1 = Dense(800, activation='relu')(inputLayer1)
    denseLayer2 = Dense(300, activation='relu')(denseLayer1)
    denseLayer3 = Dense(100, activation='relu')(denseLayer2)
    denseLayer4 = Dense(48, activation='relu')(denseLayer3)
    outputLayer1 = Dense(24, activation='softmax')(denseLayer4)

    # Model 2
    denseLayer4 = Reshape((1, 48))(denseLayer4)
    lstmLayer1 = LSTM(24, activation='tanh')(denseLayer4)
    outputLayer2 = Dense(24, activation='softmax')(lstmLayer1)

    model1 = Model(inputs = inputLayer1, outputs = outputLayer1)
    model2 = Model(inputs = inputLayer1, outputs = outputLayer2)

    return model1, model2

model1, model2 = difscorePredictModel()
model = Model(inputs = model1.inputs, outputs = model1.outputs + model2.outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

model.summary()
model.save("DifscorePredictV2.h5")