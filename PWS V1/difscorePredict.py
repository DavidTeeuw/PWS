from keras.models import load_model
import numpy as np

class difScore:
    def __init__(self) -> None:
        self.model = load_model("models/difscorePredict.keras")

    def difscorePrediction(self, boardVector: list=[]) -> int:
        prediction = self.model.predict([boardVector], verbose=0)
        return self.difScoreVal(prediction)
    
    @staticmethod
    def difScoreVal(prediction: list=[]) -> int:
        # Maximale waarde zoeken in de prediction vector
        difscore = np.argmax(prediction)

        return difscore + 10