
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QDesktopWidget, QLineEdit, QMessageBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import numpy as np
import csv
import time
import os.path
from os import path

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'TFT Tracker'
        self.left = 10
        self.top = 10
        # self.width = 320
        # self.height = 200

        self.offset = 30
        self.number_players = 7

        self.player_textbox = []
        self.start = (20, 20)
        self.widet_length = 100

        self.player_buttons = []
        self.player_dead_buttons = []
        self.text_buttons_gap = 70
        self.buttons_start = (self.start[0], self.start[1] + self.text_buttons_gap)

        # list of all dead player
        self.dead_players = np.zeros(self.number_players, dtype=int)
        # who you fought this round
        self.fought_round = np.zeros(self.number_players, dtype=int)
        # a list of the game rounds of who you played and is dead
        # first item just there so we can vstack easier
        self.all_game_rounds = np.zeros(self.number_players, dtype=int)

        # keeps track of how long you last played someone, temp AI until RNN is made and trained
        self.last_played = np.zeros(self.number_players, dtype=int) - 1
        self.last_played_labels = []

        # agjusts the mainwindow size
        self.setFixedSize(1200, 200)

        self.filename = 'tft_matches.csv'
        self.initUI()


    def initUI(self):
        self.center()
        self.setWindowTitle(self.title)

        self.initialize_textboxes()
        self.initialize_player_buttons()

        self.initialize_last_played()

        self.update_buttons = QPushButton('Update Buttons', self)
        self.update_buttons.move(self.start[0] + (self.widet_length + self.offset) * self.number_players, self.start[1]-4)
        self.update_buttons.clicked.connect(self.update_buttons_click)

        self.next_buttons = QPushButton('Next Round', self)
        self.next_buttons.move(self.start[0] + (self.widet_length + self.offset) * self.number_players, self.start[1]-4 + self.text_buttons_gap)
        self.next_buttons.clicked.connect(self.next_round_click)

        self.end_game_button = QPushButton('End Game', self)
        self.end_game_button.move(self.start[0] + (self.widet_length + self.offset) * self.number_players, self.start[1]-4 + self.text_buttons_gap*2)
        self.end_game_button.clicked.connect(self.end_game_click)
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def initialize_textboxes(self):
        for i in range(self.number_players):
            textbox = QLineEdit(self)
            textbox.move(self.start[0] + (self.widet_length + self.offset) * i, self.start[1])
            textbox.resize(self.widet_length, 20)
            # textbox.setText()
            self.player_textbox.append(textbox)

    def initialize_player_buttons(self):
        for i in range(self.number_players):
            # played button
            button = QPushButton('Player {}'.format(i+1), self)
            button.setObjectName(str(i+1))
            button.resize(self.widet_length, 30)
            button.setToolTip('Click this if you played this player')
            button.move(self.buttons_start[0] + (self.widet_length + self.offset) * i, self.buttons_start[1])
            button.clicked.connect(self.played_click)
            self.player_buttons.append(button)

            # player died button
            button = QPushButton('Died?'.format(i+1), self)
            button.setObjectName(str(i+1))
            button.resize(self.widet_length, 30)
            button.setToolTip('Click this if this player died')
            button.move(self.buttons_start[0] + (self.widet_length + self.offset) * i, self.buttons_start[1]-40)
            button.clicked.connect(self.dead_click)
            self.player_dead_buttons.append(button)

    def initialize_last_played(self):
        for i in range(self.number_players):
            label = QLabel(self)
            label.setText("-1")
            label.setObjectName(str(i+1))
            label.resize(self.widet_length, 30)
            label.move(self.buttons_start[0] + (self.widet_length + self.offset) * i, self.buttons_start[1] + 50)
            self.last_played_labels.append(label)
            label.repaint()

    @pyqtSlot()
    def update_last_played(self):
        for i in range(self.number_players):
            if self.fought_round[i] == 1:
                self.last_played[i] = 1
            elif self.last_played[i] >= 0:
                self.last_played[i] += 1
            # not elif on purpose
            if self.dead_players[i] == -1:
                self.last_played[i] = -1

            self.last_played_labels[i].setText(str(self.last_played[i]))
            self.last_played_labels[i].repaint()


    @pyqtSlot()
    def played_click(self):
        index = int(self.sender().objectName())
        self.fought_round = np.copy(self.dead_players)
        if self.fought_round[index-1] != -1:
            self.fought_round[index-1] = 1
            # forces the program tp update
            self.last_played[index-1] = 0
            self.sender().repaint()

    @pyqtSlot()
    def dead_click(self):
        index = int(self.sender().objectName())
        self.dead_players[index-1] = -1
        # forces the program tp update
        self.sender().repaint()

    @pyqtSlot()
    def update_buttons_click(self):
        for i in range(self.number_players):
            self.player_buttons[i].setText((self.player_textbox[i].text()))
            self.player_buttons[i].repaint()

    @pyqtSlot()
    def next_round_click(self):
        print(self.fought_round)
        self.all_game_rounds = np.vstack([self.all_game_rounds, self.fought_round])
        self.update_last_played()
        self.fought_round = np.copy(self.dead_players)

    @pyqtSlot()
    def end_game_click(self):
        # checks to see if the user forgot to click next round before ending the game
        if 1 in self.fought_round:
            self.all_game_rounds = np.vstack([self.all_game_rounds, self.fought_round])

        # first item just there so we can vstack easier
        print(self.all_game_rounds[1:])

        open_as = 'a'
        if not path.exists(self.filename):
            open_as = 'w'
        with open(self.filename, open_as, newline='') as csvfile:
            tft_writer = csv.writer(csvfile)
            # just using time since we are assuming not a lot of people are going to be using this
            id = int(time.time())
            # adds the game id to the row and updates the csv with game info
            for row in self.all_game_rounds[1:]:
                tft_writer.writerow(np.insert(row, 0, id))

        # set all the tracking vectors to 0
        self.fought_round = np.zeros(self.number_players, dtype=int)
        self.dead_players = np.zeros(self.number_players, dtype=int)
        self.all_game_rounds = np.zeros(self.number_players, dtype=int)
        self.last_played = np.zeros(self.number_players, dtype=int)-1
        for i in range(self.number_players):
            self.last_played_labels[i].setText(str(self.last_played[i]))
            self.last_played_labels[i].repaint()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
