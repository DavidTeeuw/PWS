import matplotlib.pyplot as plt
import numpy as np
import math
import re
import pickle

class boardManager:
    def __init__(self) -> None:
        # Pickle bestand openen
        with open('datasets/coordinates', 'rb') as f:
            self.coordinates = pickle.load(f)

        # Matrix aanmaken voor het bord
        self.board = [[0] * 35 for _ in range(38)]

    def routeNaarFrame(self, gegenereerdeRoute: list=[]) -> str:
        frame = ''

        for i in range(len(gegenereerdeRoute)):
            x, y = gegenereerdeRoute[i][0], gegenereerdeRoute[i][1]

            # Kijken welke key overeenkomt met de coördinaten
            for key, coord in self.coordinates.items():
                coord_x, coord_y = coord['x'], coord['y']
                if x == coord_x and y == coord_y:
                    frameKey = key

            # De verschillende grepen categoriseren
            if gegenereerdeRoute[i][2] == 's':
                frameToken = f'p{frameKey}r12'
            elif gegenereerdeRoute[i][2] == 'm':
                frameToken = f'p{frameKey}r13'
            elif gegenereerdeRoute[i][2] == 'e':
                frameToken = f'p{frameKey}r14'
            elif gegenereerdeRoute[i][2] == 'f':
                frameToken = f'p{frameKey}r15'
                
            frame += frameToken # De tokens bij elkaar voegen zodat het één string wordt

        return frame

    @staticmethod
    def tokenizeRoute(r: str) -> list[any]:
        patroon = r'p\d+r\d+' # Split de route na elke p####r## combinatie
        return re.findall(patroon, r)

    def frameNaarCoords(self, frame: list=[]) -> list[list[int]]:
        for i in range(len(frame)):
            # Main
            for j in range(1073, 1396):
                if str(j) in frame[i]:
                    j -= 1073
                    
                    # Coördinaten zoeken van de grepen en plaatsen in de matrix
                    if j >= 0 and j < 17:
                        rij = 36
                        kolom = (34 - (j * 2)) - 1
                    else:
                        rij = 36 - (2 * (math.trunc(j/17)))
                        kolom = 2 * (j % 17) + 1

                    if "r12" in frame[i]:
                        self.board[rij][kolom] = 1
                    elif "r13" in frame[i]:
                        self.board[rij][kolom] = 2
                    elif "r14" in frame[i]:
                        self.board[rij][kolom] = 3
                    elif "r15" in frame[i]:
                        self.board[rij][kolom] = 4
            
            # Sub
            for j in range(1447, 1600):
                if str(j) in frame[i]:
                    j -= 1447
                    
                    # Coördinaten zoeken van de grepen en plaatsen in de matrix
                    if j >= 0 and j < 18:
                        rij = 37
                        kolom = 2 * (17 - j)
                    else:
                        blok = (j - 18) // 9 # Stopt de waarde van j in blokken van 9 zodat de coördinaten overkomen met de frame
                        rij = 33 - (2 * blok)
                        
                        if blok % 2 == 0:
                            kolom = 4 * (j % 9)
                        else:
                            kolom = 2 + (4 * (j % 9))


                    if "r12" in frame[i]:
                        self.board[rij][kolom] = 1
                    elif "r13" in frame[i]:
                        self.board[rij][kolom] = 2
                    elif "r14" in frame[i]:
                        self.board[rij][kolom] = 3
                    elif "r15" in frame[i]:
                        self.board[rij][kolom] = 4

        return self.board
    
    @staticmethod
    def boardNaarIMG(board: list[list[int]]) -> plt.gcf:
        # Aanmaken van de foto
        plt.figure(figsize=(10, 10))
        boardIMG = plt.imread("imgs/board.png")

        x_as = np.array([0, 35])
        y_as = np.array([0, 39])
        plt.axis('off')

        plt.imshow(boardIMG, extent=[-1, 35, -1, 38])

        # Matrix naar coördinaten omzetten en op de foto laten zien
        x_coords = [i for i in range(35) for j in range(38) if board[j][i] != 0]
        y_coords = [37 - j for i in range(35) for j in range(38) if board[j][i] != 0]

        # De waarden in de matrix omzetten naar kleuren en plotten
        colors = [board[j][i] for i in range(35) for j in range(38) if board[j][i] != 0]
        colorMapping = {1: 'lime', 2: 'cyan', 3: 'violet', 4: 'orange'}
        edgeColors = [colorMapping[color] for color in colors]

        plt.scatter(x_coords, y_coords, facecolor='none', edgecolor=edgeColors, s=700, linewidth=2.5)
        plt.plot(x_as, y_as, linestyle='None')

        return plt.gcf()

    @staticmethod
    def boardNaarVector(board: list[list[int]]) -> list:
        # 2D array "board" veranderen in een normale array en alle waarden delen door 4
        boardVector = [greep for lijst in board for greep in lijst]
        boardVector = [i / 4 for i in boardVector]

        return boardVector
    
if __name__ == '__main__': # Voor debug
    brd = boardManager()

    route = brd.routeNaarFrame([(64, 32, 'm'), (52, 16, 's'), (48, 48, 'm'), (40, 48, 'm'), (32, 72, 'm'), (32, 88, 'm'), (16, 104, 'm'), (16, 120, 'm'), (24, 136, 'e')])

    tokens = brd.tokenizeRoute(route)
    coords = brd.frameNaarCoords(tokens)
    vector = brd.boardNaarVector(coords)

    print(vector)