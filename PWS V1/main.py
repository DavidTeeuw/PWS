from boardUtil import boardManager
from difscorePredict import difScore
from routeGenerator import routeGen
from gui import GUI

def main():
    # Methodes initialiseren
    brd = boardManager()
    difPred = difScore()
    rGen = routeGen()

    # Route genereren
    gegenereerdeRoute = rGen.genereerRoute(12)
    
    route = brd.tokenizeRoute(gegenereerdeRoute)
    board = brd.frameNaarCoords(route)
    boardVector = brd.boardNaarVector(board)

    gui = GUI(gegenereerdeRoute, difPred.difscorePrediction(boardVector), brd.boardNaarIMG(board))
    gui.run()

if __name__ == '__main__':
    main()