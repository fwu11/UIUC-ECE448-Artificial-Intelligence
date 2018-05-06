"""
    Implement your minimax and alpha-beta agents so that each of them searches to a depth of three 
    (agent moves, opponent moves, agent moves, and then evaluation function is applied to evaluate the board position). 
    Test your minimax and alpha-beta agents to make sure that, when given the same board position, 
    they both produce exactly the same move, but the alpha-beta agent comes up with that move after expanding fewer nodes in the search tree.

    Design an evaluation function that is accurate enough to permit your minimax and alpha-beta agents to beat the reflex agent. 
    Give the same evaluation function to both the minimax and alpha-beta agents. Describe your evaluation function in your report. 
"""

"""
    alpha-beta vs. minimax
    minimax vs. alpha-beta
    alpha-beta vs. reflex
    reflex vs. alpha-beta
    reflex vs. minimax
    minimax vs. reflex

"""
import numpy as np
from minimax import MINIMAX
from reflex import REFLEX
from alphabeta import ALPHABETA
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
        f = open('2.2result.txt', 'w')
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
    #red = ALPHABETA(1)
    red = REFLEX(1)
    #red = ALPHABETA(1)
    #blue = REFLEX(-1)
    blue = ALPHABETA(-1)
    board = BOARD()
    winblock = WINNINGBLOCKS()

    winblock.updateWinningBlock((3, 3), board.red_winningblock, board.blue_winningblock, 1, board.playboard)
    red.placeStone(board.playboard, (3, 3), 1)  

    '''
    add = red.getBestMove(board.playboard, 1)
    winblock.updateWinningBlock(add, board.red_winningblock, board.blue_winningblock, 1, board.playboard)
    red.placeStone(board.playboard, add, 1)
    '''
    add = blue.getBestMove(board.playboard,-1)
    winblock.updateWinningBlock(add, board.blue_winningblock, board.red_winningblock, -1, board.playboard)
    blue.placeStone(board.playboard, add, -1)

    '''
    add = red.getBestMove(board.playboard, board.red_winningblock,board.blue_winningblock, winblock, 1)
    '''
    '''
    winblock.updateWinningBlock(
        (1, 1), board.red_winningblock, board.blue_winningblock, 1, board.playboard)
    red.placeStone(board.playboard, (1, 1), 1)
    '''
    """
    winblock.updateWinningBlock((5, 5), board.blue_winningblock, board.red_winningblock, -1, board.playboard)
    blue.placeStone(board.playboard, (5, 5), -1)
    """
    """
    consider the case when there is no space-> tie
    consider 5 in a row already
    """
    while play:
        # minimax agent
        '''
        add = red.getBestMove(board.playboard, 1)
        winblock.updateWinningBlock(add, board.red_winningblock, board.blue_winningblock, 1, board.playboard)
        red.placeStone(board.playboard, add, 1)

        if board.endGame(1):
            break
        '''
        '''
        add = red.getBestMove(board.playboard, board.red_winningblock,
                              board.blue_winningblock, winblock, 1)
        '''

        # reflex agent
        add = red.getBestMove(board.red_winningblock,
                              board.blue_winningblock, board.playboard)
        winblock.updateWinningBlock(add, board.red_winningblock, board.blue_winningblock, 1, board.playboard)
        red.placeStone(board.playboard, add, 1)
        if board.endGame(1):
            break

        add = blue.getBestMove(board.playboard, -1)
        winblock.updateWinningBlock(add, board.blue_winningblock, board.red_winningblock, -1, board.playboard)
        blue.placeStone(board.playboard, add, -1)

        if board.endGame(-1):
            break
        '''
        # reflex agent
        add = blue.getBestMove(board.blue_winningblock,board.red_winningblock, board.playboard)
        winblock.updateWinningBlock(add, board.blue_winningblock, board.red_winningblock, -1, board.playboard)
        blue.placeStone(board.playboard, add, -1)
        if board.endGame(-1):
            break
        '''
    board.printResult()
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
