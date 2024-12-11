from keras.models import load_model
import numpy as np
import pandas as pd

from boardUtil import boardManager

class routeGen:
    def __init__(self) -> None:
        self.df = pd.read_csv(r'datasets/dataset_frames_difscore.csv', header=None, names=['routes', 'difscore'])
        self.df['routes'] = self.df['routes'].apply(boardManager.tokenizeRoute)
        self.model = load_model("models/routeGen.keras")
        self.vocab, self.tokenNaarIdx, self.idxNaarToken = self.maakVocab()
        self.max_route_lengte = max(len(route) for route in self.df['routes'])

    def maakVocab(self) -> tuple[list, dict[any, int], dict[int, any]]:
        # Alle mogelijke grepen een index geven
        vocab = sorted(set(token for route in self.df['routes'] for token in route))
        vocab.append('<PAD>')
        tokenNaarIdx = {token: idx for idx, token in enumerate(vocab)} 
        idxNaarToken = {idx: token for token, idx in tokenNaarIdx.items()} # Om van index waarden terug te gaan naar tokens

        return vocab, tokenNaarIdx, idxNaarToken
    
    def genereerRoute(self, maxRouteLengte: int) -> str: # Een maximale lengte van de routes om te voorkomen dat hij een route genereerd met teveel grepen
        while True:    
            # Pak een startgreep van een willekeurige route uit de dataset als eerste greep voor de route
            randomRoute = self.df['routes'].sample().iloc[0]
            generated = [self.tokenNaarIdx[token] for token in randomRoute[:1]]

            # Nieuwe route genereren
            for _ in range(maxRouteLengte):
                prediction = self.model.predict(np.array([generated + [self.tokenNaarIdx['<PAD>']] * (self.max_route_lengte - len(generated))]), verbose=0)
                volgende_greep = np.argmax(prediction[0, len(generated)-1, :])      
                if volgende_greep == self.tokenNaarIdx['<PAD>']:
                    break
                generated.append(volgende_greep)

            # Controleren of de route voldoet aan maxRouteLengte
            if (generated[-1] != self.tokenNaarIdx['<PAD>'] and len(generated) <= maxRouteLengte):
                gegeneerdeRoute =  [self.idxNaarToken[idx] for idx in generated]
                return ''.join(gegeneerdeRoute)