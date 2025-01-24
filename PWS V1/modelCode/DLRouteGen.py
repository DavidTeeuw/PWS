import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd # type: ignore
from tensorflow.keras import layers, models # type: ignore
import numpy as np # type: ignore
import re

# Data laden
df = pd.read_csv(r'dataset_frames_difscore.csv', header=None, names=['routes', 'difscore'])

# Alle routes tokenizen
def tokenizeRoute(r):
    pattern = r'p\d+r\d+' # Split de route na elke p####r## combinatie
    return re.findall(pattern, r)

df['routes'] = df['routes'].apply(tokenizeRoute)

# Alle mogelijke grepen een index geven
vocab = sorted(set(token for route in df['routes'] for token in route))
vocab.append('<PAD>')
tokenNaarIdx = {token: idx for idx, token in enumerate(vocab)}

numTokens = len(tokenNaarIdx) # Aantal verschillende tokens in de index

# Array X en Y aanmaken die even lang is als de langste route in de dataset
maxRouteLengte = max(len(route) for route in df['routes'])
X = np.zeros((len(df), maxRouteLengte), dtype=int)
y = np.zeros((len(df), maxRouteLengte), dtype=int)

for i, route in enumerate(df['routes']):
    X[i, :len(route)] = [tokenNaarIdx[token] for token in route] # Elke greep van een route uit de dataset in X plaatsen
    y[i, :len(route)-1] = [tokenNaarIdx[token] for token in route[1:]] # Elke volgende greep van de route uit de dataset in Y plaatsen
    if len(route) < maxRouteLengte: # Padding toevoegen als de route te kort is
        X[i, len(route):] = tokenNaarIdx['<PAD>']
        y[i, len(route)-1:] = tokenNaarIdx['<PAD>']

# Model architectuur
def routeGenModel(maxRouteLengte, numTokens):
    model = models.Sequential([
        layers.Embedding(numTokens, 64, input_length=maxRouteLengte),
        layers.LSTM(128, return_sequences=True),
        layers.Dense(numTokens, activation='softmax')
    ])
    return model

# Model trainen
model = routeGenModel(maxRouteLengte, numTokens)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Model opslaan
model.summary()
model.save('routeGen.keras')