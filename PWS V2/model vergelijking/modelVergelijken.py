from keras.models import load_model
import numpy as np
import pandas as pd

df = pd.read_csv(r'datasets/dataset_frames_vector_difscore.csv', header=None, names=['vectors', 'difscore'])
df['vectors'] = df['vectors'].apply(lambda x: np.array(list(map(float, x.split()))))

model = load_model("models/DifscorePredictV1.h5") # Model 1
#model = load_model("models/DifscorePredictV2.h5") # Model 2

def difscoreVal(prediction: list=[]) -> int:
    # Pak de index met de grootste waarde
    top_index_1 = np.argsort(prediction[0][0])[-1:]
    top_index_2 = np.argsort(prediction[1][0])[-1:]

    # Pak het gemiddelde van de twee indexwaarden
    avg_index = (np.mean(top_index_1) + np.mean(top_index_2)) / 2

    # Afronden en +10 omdat de difscore van 10-33 is
    difscore = round(avg_index) + 10

    return difscore

for i in range(10): # Aantal keer dat je het model wil testen
    j = 0

    routes = df.sample(n=100) # 100 willekeurige routes pakken uit de dataset

    data = {
        "predicted grade": [0] * 100,
        "actual grade": [0] * 100,
        "verschil": [0] * 100
    }

    for index, route in routes['vectors'].items():
        # Array de juist vorm geven zodat het gebruikt kan worden
        route = np.array(route).reshape(1, -1)
        prediction = model.predict([route], verbose=0)

        difscore = round(np.argmax(prediction) + 10) # Deze voor model 1
        #difscore = difscoreVal(prediction) # Deze voor model 2
        actual_difscore = df.loc[index, 'difscore']

        # Data toevoegen aan de dataset
        data["predicted grade"][j] = difscore
        data["actual grade"][j] = actual_difscore
        data["verschil"][j] = abs(difscore - actual_difscore)

        j += 1

    modelScoreDataFrame = pd.DataFrame(data)

    modelScoreDataFrame.to_excel(f"model vergelijking/model 1/model1Score{i}.xlsx") # Deze voor model 1
    #modelScoreDataFrame.to_excel(f"model vergelijking/model 2/model2Score{i}.xlsx") # Deze voor model 2