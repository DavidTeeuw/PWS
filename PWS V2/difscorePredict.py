from keras.models import load_model
import numpy as np

class difScore:
    def __init__(self) -> None:
        self.model = load_model("models/DifscorePredictV2.h5")

    def difscorePrediction(self, boardVector: list=[]) -> int:
        prediction = self.model.predict([boardVector], verbose=0)
        return self.difscoreVal(prediction)
    
    @staticmethod
    def difscoreVal(prediction: list=[]) -> int:
        # Pak de index met de grootste waarde
        top_index_1 = np.argsort(prediction[0][0])[-1:]
        top_index_2 = np.argsort(prediction[1][0])[-1:]

        # Pak het gemiddelde van de twee indexwaarden
        avg_index = (np.mean(top_index_1) + np.mean(top_index_2)) / 2

        # Afronden en +10 omdat de difscore van 10-33 is
        difscore = round(avg_index) + 10

        return difscore
