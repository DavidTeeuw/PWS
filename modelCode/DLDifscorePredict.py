import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras import layers, models # type: ignore
import numpy as np # type: ignore

# Data ophalen en klaarmaken om het model te trainen
df = pd.read_csv(r'datasets/dataset_frames_vector_difscore.csv', header=None, names=['vectors', 'difscore'])
df['vectors'] = df['vectors'].apply(lambda x: np.array(list(map(float, x.split()))))

X = np.array(df['vectors'].tolist())
y = df['difscore'] - 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architectuur
def difscorePredictModel():
    model = models.Sequential([
        layers.Flatten(input_shape=(1330,)),
        layers.Dense(800, activation='relu'),
        layers.Dense(300, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(48, activation='relu'),
        layers.Dense(24, activation='softmax')
    ])
    return model

model = difscorePredictModel()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=32, validation_split=0.2)

model.summary()
model.save('DLDifscorePredict.keras')