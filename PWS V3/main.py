from boardUtil import boardManager
from difscorePredict import difScore
from routeGenerator import routeGen

import matplotlib.pyplot as plt

def main():
    # Methodes initialiseren
    brd = boardManager()
    difPred = difScore()
    rGen = routeGen()

    # Route genereren
    gegenereerdeRoute = rGen.genereerRoute(36)
    routeFrame = brd.routeNaarFrame(gegenereerdeRoute)
    
    route = brd.tokenizeRoute(routeFrame)
    board = brd.frameNaarCoords(route)
    boardVector = brd.boardNaarVector(board)

    print("Route: ", routeFrame)
    print("Difscore: ", difPred.difscorePrediction(boardVector))

    brd.boardNaarIMG(board)

    plt.show()

if __name__ == '__main__':
    main()