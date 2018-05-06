import numpy as np

class REFLEX():
    def __init__(self, side):
        # (x,y) = (horizontal,vertical)
        #self.red = (1,1)
        #self.blue = (5,5)

        self.counter = side

    def breakATie(self, loc1, loc2):
        if loc1[0] < loc2[0] or (loc1[0] == loc2[0] and loc1[1] < loc2[1]):
            return loc1
        else:
            return loc2

    def checkNeighbour(self, vec):
        """
        set impossible candidate to 1
        """
        tmp = np.asarray(vec)
        # left shift
        left_shift = np.roll(tmp, -1)
        left_shift[-1] = 0
        # right shift
        right_shift = np.roll(tmp, 1)
        right_shift[0] = 0

        result = tmp + left_shift + right_shift

        return tuple(tmp + (result == 0))

    def decodeLocation(self, vec, start, end):
        """
        return a list of coordinates to explore
        """

        dx = (end[0] - start[0]) // 4
        dy = (end[1] - start[1]) // 4
        step = np.array([0, 1, 2, 3, 4])
        step.astype(int)
        x = start[0] + dx * step
        y = start[1] + dy * step
        idx = np.asarray(vec)
        x = x[idx == 0]
        y = y[idx == 0]
        return [(x[i], y[i]) for i in range(len(x))]

    def placeStone(self, board, add, side):
        if side == 1:
            board[add[1]][add[0]] = self.counter
            self.counter += 1
            print('red stone location' +
                  '(' + str(add[0]) + ',' + str(add[1]) + ')')
        elif side == -1:
            board[add[1]][add[0]] = self.counter
            self.counter -= 1
            print('blue stone location' +
                  '(' + str(add[0]) + ',' + str(add[1]) + ')')

    def secondPriority(self, opponent_winningblock):
        if 4 in opponent_winningblock:
            for i in range(len(opponent_winningblock[4])):
                vec = opponent_winningblock[4][i][2]
                if vec[4] == 0 or vec[0] == 0:
                    return True
        return False

    def thirdPriority(self, opponent_winningblock):
        if 3 in opponent_winningblock:
            for i in range(len(opponent_winningblock[3])):
                vec = opponent_winningblock[3][i][2]
                if vec[0] == 0 and vec[4] == 0:
                    return True

        return False

    def getBestMove(self, player_winningblock, opponent_winningblock, board):
        # winning block of the form {length:[((start_x,start_y),(end_x,end_y),(1,1,1,1,1)),(...)}
        """
        Need to implement when there is no winning block
        """
        add = None
        # Check whether the agent side is going to win
        # by placing just one more stone.
        if 4 in player_winningblock:
            temp = (6, 6)
            for i in range(len(player_winningblock[4])):
                candidates = self.decodeLocation(
                    player_winningblock[4][i][2], player_winningblock[4][i][0], player_winningblock[4][i][1])
                temp = self.breakATie(candidates[0], temp)
            add = temp
            return add

        elif self.secondPriority(opponent_winningblock):
            # Then check whether the opponent has an UNBROKEN chain formed by 4 stones
            # and has an empty intersection on either head of the chain.
            temp = (6, 6)
            for i in range(len(opponent_winningblock[4])):
                vec = opponent_winningblock[4][i][2]
                if vec[4] == 0 or vec[0] == 0:
                    candidates = self.decodeLocation(
                        vec, opponent_winningblock[4][i][0], opponent_winningblock[4][i][1])
                    temp = self.breakATie(candidates[0], temp)
            add = temp

        elif self.thirdPriority(opponent_winningblock):
            # Check whether the opponent has an UNBROKEN chain formed by 3 stones and
            # has empty spaces on both ends of the chain.
            temp = (6, 6)
            for i in range(len(opponent_winningblock[3])):
                vec = opponent_winningblock[3][i][2]
                if vec[0] == 0 and vec[4] == 0:
                    candidates = self.decodeLocation(
                        vec, opponent_winningblock[3][i][0], opponent_winningblock[3][i][1])
                    for j in range(2):
                        temp = self.breakATie(candidates[j], temp)
            add = temp

        else:
            # find all possible sequences of 5 consecutive spaces that contain none of the opponent's stones
            # find the winning block which has the largest number of the agent stones
            # Last, in the winning block, place a stone next to a stone already in the winning block on board.

            # check if there is no more winning block
            if len(player_winningblock) == 0:
                for j in range(7):
                    for i in range(7):
                        if board[j][i] == 0:
                            add = (i, j)
                            return add
            else:
                idx = max(player_winningblock.keys())
                temp = (6, 6)

                for i in range(len(player_winningblock[idx])):
                    valid = self.checkNeighbour(player_winningblock[idx][i][2])
                    candidates = self.decodeLocation(
                        valid, player_winningblock[idx][i][0], player_winningblock[idx][i][1])
                    for j in range(5 - int(sum(valid))):
                        temp = self.breakATie(candidates[j], temp)
                add = temp
        return add
