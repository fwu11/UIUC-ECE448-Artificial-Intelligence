import numpy as np
from alphabeta import ALPHABETA
from stochastic import STOCHASTIC
from winningblocks import WINNINGBLOCKS
import time

class BOARD():
    def __init__(self):
        self.playboard = np.zeros((7, 7))
        # winning block with the format of {length:[((start_x,start_y),(end_x,end_y),(1,1,1,1,1)),(...)}
        self.red_winningblock = dict()
        self.blue_winningblock = dict()

    def endGame(self, side):
        # check if it is a tie or someone wins
        if side == 1:
            if 5 in self.red_winningblock:
                print('RED WIN')
                return True
            elif not 0 in self.playboard:
                print('TIE GAME')
                return True
        elif side == -1:
            if 5 in self.blue_winningblock:
                print('BLUE WIN')
                return True
            elif not 0 in self.playboard:
                print('TIE GAME')
                return True
        return False

    def printResult(self):
        """
        '.' to denote empty intersections
        small letters to denote each of the red stones (first player)
        capital letters to denote each of the blue stones (second player)  

        """
        f = open('2.3result.txt', 'w')
        for j in reversed(range(7)):
            for i in range(7):
                if self.playboard[j][i] == 0:
                    f.write('.')
                elif self.playboard[j][i] > 0:
                    f.write(chr(ord('a') + int(self.playboard[j][i]) - 1))
                elif self.playboard[j][i] < 0:
                    f.write(chr(ord('A') + abs(int(self.playboard[j][i])) - 1))
            f.write('\n')
        f.close()


def main():
    start = time.time()
    # side = 1 -> red
    # side = -1 -> blue
    play = True
    red = STOCHASTIC(1)
    blue = ALPHABETA(-1)
    board = BOARD()
    winblock = WINNINGBLOCKS()

    while play:

        add = red.getBestMove(board.playboard, 1)
        winblock.updateWinningBlock(add, board.red_winningblock, board.blue_winningblock, 1, board.playboard)
        red.placeStone(board.playboard, add, 1)

        if board.endGame(1):
            break

        add = blue.getBestMove(board.playboard, -1)
        winblock.updateWinningBlock(add, board.blue_winningblock, board.red_winningblock, -1, board.playboard)
        blue.placeStone(board.playboard, add, -1)

        if board.endGame(-1):
            break

    board.printResult()
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    #import cProfile
    #cProfile.run('main()')
    main()