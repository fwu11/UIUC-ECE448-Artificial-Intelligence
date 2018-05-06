# implement MINIMAX
# fwu11

import numpy as np


class MOVE():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.score = None


class MINIMAX():
    def __init__(self, side):
        self.counter = side
        self.node_expand = 0

    def evaluation(self,side,board):
        score = 0
        pos_shift = np.array([-4, -3, -2, -1, 0])
        neg_shift = np.array([4, 3, 2, 1, 0])
        zero_shift = np.array([0,0,0,0,0])
        order = [(pos_shift,zero_shift),(zero_shift,pos_shift),(pos_shift,pos_shift),(pos_shift,neg_shift)]
        increment = [(np.array([0,1,2,3,4]),np.array([0,0,0,0,0])),(np.array([0,0,0,0,0]),np.array([0,1,2,3,4])),
        (np.array([0,1,2,3,4]),np.array([0,1,2,3,4])),(np.array([0,1,2,3,4]),np.array([0,-1,-2,-3,-4]))]

        _,player_move,opponent_move= self.decodeBoard(board, side)

        for idx in player_move:
            x = idx[0]
            y = idx[1]

            for i in range(4):
                x_val = x + order[i][0]
                y_val = y + order[i][1]
                for j in range(5):
                    mask_x = x_val[j]+increment[i][0]
                    mask_y = y_val[j]+increment[i][1]

                    # out of bound -> cannot be a winning block
                    if (mask_x[4] >= 7 or mask_x[4] < 0) or (mask_y[4] >= 7 or mask_y[4]< 0) or (mask_x[0] >= 7 or mask_x[0] < 0) or (mask_y[0] >= 7 or mask_y[0]< 0):
                        continue

                    loc = list(zip(mask_x, mask_y))
                    block = [board[int(n)][int(m)] for (m, n) in loc]
                    player = np.array(block) * side > 0
                    opponent = np.array(block) * -side > 0
                    if not (True in opponent): 
                        score += 10**(sum(player))

        for idx in opponent_move:
            x = idx[0]
            y = idx[1]

            for i in range(4):
                x_val = x + order[i][0]
                y_val = y + order[i][1]
                for j in range(5):
                    mask_x = x_val[j]+increment[i][0]
                    mask_y = y_val[j]+increment[i][1]

                    # out of bound -> cannot be a winning block
                    if (mask_x[4] >= 7 or mask_x[4] < 0) or (mask_y[4] >= 7 or mask_y[4]< 0) or (mask_x[0] >= 7 or mask_x[0] < 0) or (mask_y[0] >= 7 or mask_y[0]< 0):
                        continue

                    loc = list(zip(mask_x, mask_y))
                    block = [board[int(n)][int(m)] for (m, n) in loc]
                    player = np.array(block) * side > 0
                    opponent = np.array(block) * side < 0
                    if not(True in player): 
                        if sum(opponent) == 5 or sum(opponent)==4 or sum(opponent) == 3:        
                            score -= 10**(sum(opponent)+1)
                        else:
                            score -= 10**(sum(opponent))
        return score

    def decodeBoard(self, board, side):
        empty_space = np.where(board == 0)
        player = np.where(side * board > 0)
        opponent = np.where(side*board < 0)
        return list(zip(empty_space[1],empty_space[0])),list(zip(player[1],player[0])),list(zip(opponent[1],opponent[0]))

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

    def findBestMove(self, moves, flag):
        bestMove = None
        if flag == 'max':
            bestMove = max(moves, key=lambda MOVE: MOVE.score)
        elif flag == 'min':
            bestMove = min(moves, key=lambda MOVE: MOVE.score)

        return bestMove

    def getBestMove(self, board, side):
        """
            minimax with depth of 3: Max Min MAX
            return the coordinates of the best move and a flag shows if it is the end of the game
            in getMaxValue / getMinValue need to consider how to inherent the score from children
        """

        self.node_expand = 0
        # call the recursive function
        bestMove = self.getMaxValue(board, 3 - 1, side)

        # clear the winblock before return

        if side == 1:
            print('red node expand:' + str(self.node_expand))
        else:
            print('blue node expand:' + str(self.node_expand))

        return (bestMove.x, bestMove.y)

    def getMaxValue(self, board, depth, side):
        # player side
        # get all its childern's moves and get the best score
        moves = []
        empty_space,_,_=self.decodeBoard(board, side)
        for idx in range(len(empty_space)):
            node = MOVE(empty_space[idx][0], empty_space[idx][1])
            board[node.y][node.x] = side
            self.node_expand += 1
            # only one left or at depth of 3
            if len(empty_space) == 1 or depth == 0:
                node.score = self.evaluation(side, board)
            else:
                result = self.getMinValue(board, depth - 1, side)
                node.score = result.score
            moves.append(node)
            board[node.y][node.x] = 0

        bestMove = self.findBestMove(moves, 'max')

        return bestMove

    def getMinValue(self, board, depth, side):
        # for next level, first replace board with that value then
        # clear that value
        # opponent -> min
        moves = []
        empty_space,_,_=self.decodeBoard(board, side)
        for idx in range(len(empty_space)):
            node = MOVE(empty_space[idx][0], empty_space[idx][1])
            board[node.y][node.x] = -side
            self.node_expand += 1
            # only one left or at depth of 3
            if len(empty_space) == 1 or depth == 0:
                node.score = self.evaluation(side, board)
            else:
                result = self.getMaxValue(board, depth - 1, side)
                node.score = result.score
            moves.append(node)
            board[node.y][node.x] = 0

        bestMove = self.findBestMove(moves, 'min')

        return bestMove
