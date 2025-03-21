from keras.models import load_model
import numpy as np
import pickle
import random

class routeGen:
    def __init__(self) -> None:
        # Pickle bestand openen
        with open('datasets/merged_dataset', 'rb') as f:
            routes = pickle.load(f)

        routeList = list(routes.items())
        
        # Data filteren
        cutoffGrade  = ['8b/V13', '8b+/V14', '8c/V15', '8c+/V16', '9a/V17', '9a+/V18', '9b/V19', '9b+/V20', '9c/V21', '9c+/V22']
        self.gefilterdeRoutes = [route for route in routeList if route[1]['grade_name'] not in cutoffGrade and len(route[1]['frames']) < 36]        

        self.model = load_model("models/RouteGenV2.h5")
        self.vocab, self.tokenNaarIdx, self.idxNaarToken = self.maakVocab()
        self.max_route_lengte = max(len(route[1]['frames']) for route in self.gefilterdeRoutes)

    def maakVocab(self) -> tuple[list, dict[any, int], dict[int, any]]:
        # Alle mogelijke grepen een index geven
        vocab = sorted(set(tuple(frame) for _, routes in self.gefilterdeRoutes for frame in routes['frames']))
        vocab.append('<PAD>')
        tokenNaarIdx = {token: idx for idx, token in enumerate(vocab)}
        idxNaarToken = {idx: token for token, idx in tokenNaarIdx.items()} # Om van index waarden terug te gaan naar tokens

        return vocab, tokenNaarIdx, idxNaarToken
    
    def genereerRoute(self, maxRouteLengte: int) -> list[tuple]: # Een maximale lengte van de routes om te voorkomen dat hij een route genereerd met teveel grepen
        while True:    
            # Pak een startgreep van een willekeurige route uit de dataset als eerste greep voor de route
            randomRoute = random.choice(self.gefilterdeRoutes)
            startFrame = randomRoute[1]['frames'][0]
            route = [self.tokenNaarIdx[tuple(startFrame)]]

            willekeurigheidsFactor = 5 # Bepaald de top hoeveel predictions waar de generator uit kan kiezen

            # Nieuwe route genereren
            for _ in range(maxRouteLengte - 1):
                prediction = self.model.predict(np.array([route + [self.tokenNaarIdx['<PAD>']] * (self.max_route_lengte - len(route))]), verbose=0)
                volgendeGreep = np.argmax(prediction[0, len(route) - 1, :])
    
                topPredictions = np.argsort(prediction[0, len(route) - 1, :])[-willekeurigheidsFactor:][::-1] # Pak de top drie grepen

                if volgendeGreep == self.tokenNaarIdx['<PAD>']:
                    break
                route.append(topPredictions[random.randint(0, willekeurigheidsFactor - 1)]) # Voeg een van de top drie grepen willekeurig toe

            # Controleren of de route voldoet aan maxRouteLengte
            if len(route) <= maxRouteLengte:
                gegenereerdeRoute = [self.idxNaarToken[idx] for idx in route]
                return gegenereerdeRoute

if __name__ == "__main__": # Voor debug
    rGen = routeGen()
    print(rGen.genereerRoute(12))