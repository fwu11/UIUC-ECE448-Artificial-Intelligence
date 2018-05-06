import numpy as np
import copy

class WINNINGBLOCKS():

    def updateWinningBlock(self, add, current_winningblock, opponent_winningblock,side,board):
        # update after the current player place the stone and before the next player play
        # update the current player winning block 
        # then update the opponent winning block

        #current_winningblock = copy.deepcopy(pl_winblock)
        #opponent_winningblock = copy.deepcopy(op_winblock)
        
        pos_shift = np.array([-4, -3, -2, -1, 0])
        neg_shift = np.array([4, 3, 2, 1, 0])
        zero_shift = np.array([0,0,0,0,0])
        x = add[0]
        y = add[1]
        order = [(pos_shift,zero_shift),(zero_shift,pos_shift),(pos_shift,pos_shift),(pos_shift,neg_shift)]
        increment = [([0,1,2,3,4],[0,0,0,0,0]),([0,0,0,0,0],[0,1,2,3,4]),([0,1,2,3,4],[0,1,2,3,4]),([0,1,2,3,4],[0,-1,-2,-3,-4])]

        # winning block with the format of {length:[((start_x,start_y),(end_x,end_y),(1,1,1,1,1)),(...)}

        # horizontally            
        # vertically
        # diagonally
        # anti-diagonally
        for idx in range(4):
            x_val = x + order[idx][0]
            y_val = y + order[idx][1]

            for i in range(5):
                for j in range(5):
                    flag = False
                    # out of bound -> cannot be a winning block
                    if (x_val[i] + increment[idx][0][j] >= 7 or x_val[i] + increment[idx][0][j] < 0) or (y_val[i] + increment[idx][1][j] >= 7 or y_val[i] + increment[idx][1][j] < 0):
                        break
                    # opponent in the proposed winning block -> do not consider
                    if self.checkOpponent(x_val[i]+increment[idx][0][j], y_val[i]+increment[idx][1][j], side,board):
                        break
                    # can be a winning block
                    if j == 4:
                        item_to_delete = []
                        # if previously a existed winning block -> add element to it
                        for key, value in current_winningblock.items():
                            for k in range(len(value)):
                                if self.sameWinningBlock(value[k][0], value[k][1],(x_val[i],y_val[i]),(x_val[i]+increment[idx][0][j], y_val[i]+increment[idx][1][j])):
                                    tmp = np.asarray(value[k][2])
                                    tmp[self.findPosition(value[k][0],value[k][1],add)]=1
                                    item_to_delete.append((key,value[k],tmp))
                                    flag == True
                                    break
                            if flag == True:
                                break
                        # update dict
                        for itr in range(len(item_to_delete)):    
                            key = item_to_delete[itr][0] 
                            element = item_to_delete[itr][1]
                            tmp = item_to_delete[itr][2]   
                            if element in current_winningblock[key]:
                                current_winningblock[item_to_delete[itr][0]].remove(element)
                                if len(current_winningblock[key]) == 0:
                                    current_winningblock.pop(key,None)
                            current_winningblock.setdefault(key+1,[]).append((element[0],element[1],tuple(tmp)))
                        
                        if flag == False:
                            # else, new winning block -> create it
                            tmp = np.zeros(5)
                            tmp[self.findPosition((x_val[i],y_val[i]),(x_val[i]+increment[idx][0][j], y_val[i]+increment[idx][1][j]),add)] = 1
                            current_winningblock.setdefault(1, []).append(((x_val[i],y_val[i]),(x_val[i]+increment[idx][0][j], y_val[i]+increment[idx][1][j]),tuple(tmp)))


        # check if add affects the opponent winning block
        # then delete that winning block
        item_to_delete = []
        for key, value in opponent_winningblock.items():
            for k in range(len(value)):
                if self.checkInBetween(add, value[k][0], value[k][1]):
                    item_to_delete.append((key,value[k]))

        for itr in range(len(item_to_delete)):
            if item_to_delete[itr][1] in opponent_winningblock[item_to_delete[itr][0]]:
                opponent_winningblock[item_to_delete[itr][0]].remove(item_to_delete[itr][1])
                if len(opponent_winningblock[item_to_delete[itr][0]]) == 0:
                    opponent_winningblock.pop(item_to_delete[itr][0],None)

        #return current_winningblock,opponent_winningblock


    def checkInBetween(self, loc, start, end):

        dx = (end[0]-start[0])//4
        dy = (end[1]-start[1])//4
        step = np.array([0,1,2,3,4])
        x = start[0] + dx*step
        y = start[1] + dy*step

        return loc in list(zip(x.tolist(),y.tolist()))

    def checkOpponent(self, x, y, side,board):
        # given side, decide if opponent on the way
        if side == 1:
            return board[int(y)][int(x)] < 0
        elif side == -1:
            return board[int(y)][int(x)] > 0

    def sameWinningBlock(self,start,end,begin,stop):
        return (start == begin and end == stop) or (start == stop and end == begin)

    def findPosition(self,start,end,loc):
        dx = (end[0]-start[0])//4
        dy = (end[1]-start[1])//4
        step = np.array([0,1,2,3,4])
        x = start[0] + dx*step
        y = start[1] + dy*step
        return list(zip(x.tolist(),y.tolist())).index(loc)
   