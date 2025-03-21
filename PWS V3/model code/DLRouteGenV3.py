# Project gestopt wegens tekort aan tijd
# Dit script is niet af en ook nooit gebruikt

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

routeGradeBuckets = [['4a/V0', '4b/V0', '4c/V0'], 
                     ['5a/V1', '5b/V1', '5c/V2'], 
                     ['6a/V3', '6a+/V3'], 
                     ['6b/V4', '6b+/V4'], 
                     ['6c/V5', '6c+/V5'], 
                     ['7a/V6', '7a+/V7'], 
                     ['7b/V8', '7b+/V8'], 
                     ['7c/V9', '7c+/V10'],
                     ['8a/V11', '8a+/V12']]

# Create a dictionary to hold the routes for each grade bucket
bucketedRoutes = {tuple(bucket): [] for bucket in routeGradeBuckets}

# Sort the routes into the appropriate grade buckets
for route in gefilterdeRoutes:
    grade_name = route[1]['grade_name']
    for bucket in routeGradeBuckets:
        if grade_name in bucket:
            bucketedRoutes[tuple(bucket)].append(route)
            break

# Access the first bucket
first_bucket_key, first_bucket_routes = next(iter(bucketedRoutes.items()))

# Print the key and the routes in the first bucket
print("First bucket key:", first_bucket_key)
#print("Routes in the first bucket:", first_bucket_routes)

vocab = sorted(set(tuple(frame) for _, routes in gefilterdeRoutes for frame in routes['frames']))
vocab.append('<PAD>')
tokenNaarIdx = {token: idx for idx, token in enumerate(vocab)}

numTokens = len(tokenNaarIdx) # Aantal verschillende tokens in de index

def routeGenModel(maxRouteLengte, numTokens):
    model = models.Sequential([
        layers.Embedding(numTokens, 64, input_length=maxRouteLengte),
        layers.LSTM(128, return_sequences=True),
        layers.Dense(numTokens, activation='softmax')
    ])

    return model

for bucket in bucketedRoutes:
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
