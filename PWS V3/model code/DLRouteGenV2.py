import pickle
import sys
import numpy as np
from tensorflow.keras import layers, models # type: ignore

sys.stdout.reconfigure(encoding='utf-8') # Om errors te voorkomen

# Pickle bestanden openen
with open('datasets/merged_dataset', 'rb') as f:
    routes = pickle.load(f)

routeList = list(routes.items())

# Filter de routes zodat er geen slechte routes in zitten
# In de hoge grades zitten routes van mensen die niet serieus bezig waren met het maken van de routes
# Daarom niet trainen op de hoge grades
# Er zijn ook routes van mensen die gewoon alle grepen op het bord hebben geselecteerd
# Dat is geen goede data en is niet handig om op te trainen, daarom alle routes kiezen die minder dan 36 grepen hebben
cutoffGrade  = ['8b/V13', '8b+/V14', '8c/V15', '8c+/V16', '9a/V17', '9a+/V18', '9b/V19', '9b+/V20', '9c/V21', '9c+/V22']
gefilterdeRoutes = [route for route in routeList if route[1]['grade_name'] not in cutoffGrade and len(route[1]['frames']) < 36]

vocab = sorted(set(tuple(frame) for _, routes in gefilterdeRoutes for frame in routes['frames']))
vocab.append('<PAD>')
tokenNaarIdx = {token: idx for idx, token in enumerate(vocab)}

numTokens = len(tokenNaarIdx) # Aantal verschillende tokens in de index

# Array X en Y aanmaken die even lang is als de langste route in de dataset
maxRouteLengte = max(len(route[1]['frames']) for route in gefilterdeRoutes)
X = np.zeros((len(gefilterdeRoutes), maxRouteLengte), dtype=int)
y = np.zeros((len(gefilterdeRoutes), maxRouteLengte), dtype=int)

# Vul X en y
for i, (_, route_data) in enumerate(gefilterdeRoutes):
    frames = route_data['frames']
    frameIndices = [tokenNaarIdx[tuple(frame)] for frame in frames]  # Zet grepen om naar indices

    X[i, :len(frames)] = frameIndices # Elke greep van een route uit de dataset in X plaatsen
    y[i, :len(frames)-1] = frameIndices[1:] # Elke volgende greep van de route uit de dataset in Y plaatsen
    
    # Padding indien nodig
    if len(frames) < maxRouteLengte:
        X[i, len(frames):] = tokenNaarIdx['<PAD>']
        y[i, len(frames)-1:] = tokenNaarIdx['<PAD>']

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
model.fit(X, y, epochs=1, batch_size=32, validation_split=0.2)

# Model opslaan
model.summary()
model.save('models/RouteGenV2.h5')