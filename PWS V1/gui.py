import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from routeGenerator import routeGen
from difscorePredict import difScore
from boardUtil import boardManager

class GUI:
    def __init__(self, route, difscore, figure) -> None:
        self.route = route
        self.difscore = difscore
        self.window = tk.Tk()
        self.figure = figure
        self.window.title("Kilterboard AI")
        self.window.state('zoomed')

    def app(self) -> None:
        self.window.update_idletasks()

        # De app in twee delen opdelen
        self.routeFrame = tk.Frame(self.window)
        self.restFrame = tk.Frame(self.window)

        # De twee delen aanmaken
        self.routeFrame.pack(fill="both", expand=True)
        self.restFrame.pack(fill="both", expand=True)

        # Proporties instellen
        self.routeFrame.pack_propagate(False)
        self.restFrame.pack_propagate(False)

        # Hoogte van elk frame instellen
        self.wHeight = self.window.winfo_height()
        self.routeFrame.config(height=int(self.wHeight * 0.9))
        self.restFrame.config(height=int(self.wHeight * 0.1))

        self.boardCanvas = FigureCanvasTkAgg(self.figure, master=self.routeFrame)
        self.boardCanvas.draw()
        self.boardCanvas.get_tk_widget().pack()

        button = tk.Button(self.restFrame, text="Nieuwe Route", command=self.knopActie)
        button.pack()

        # Labels aanmaken om de gegevens van de route te printen
        self.routeLabel = tk.Label(self.restFrame, text=f"Route: {''.join(self.route)}")
        self.difscoreLabel = tk.Label(self.restFrame, text=f"Difficulty Score: {self.difscore}")
        self.routeLabel.pack()
        self.difscoreLabel.pack()

    def knopActie(self) -> None:
        # Methodes initialiseren
        rGen = routeGen()
        difPred = difScore()
        brd = boardManager()

        # Route genereren en difscore bepalen
        self.gegeneerdeRoute = rGen.genereerRoute(12)

        self.route = brd.tokenizeRoute(self.gegeneerdeRoute)
        self.board = brd.frameNaarCoords(self.route)
        self.boardVector = brd.boardNaarVector(self.board)

        self.difscore = difPred.difscorePrediction(self.boardVector)

        # Labels en figure updaten
        self.routeLabel.config(text=f"Route: {''.join(self.gegeneerdeRoute)}")
        self.difscoreLabel.config(text=f"Difficulty Score: {self.difscore}")

        self.boardCanvas.get_tk_widget().destroy()

        self.boardCanvas = FigureCanvasTkAgg(brd.boardNaarIMG(self.board), master=self.routeFrame)
        self.boardCanvas.draw()
        self.boardCanvas.get_tk_widget().pack()

    def run(self) -> None:
        self.app()
        self.window.bind('<f>', lambda _: self.window.attributes("-fullscreen", not self.window.attributes("-fullscreen")))
        self.window.bind('<Escape>', lambda _: self.window.attributes("-fullscreen", False))
        self.window.mainloop()