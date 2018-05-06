# implement the Reflex agent
# fwu11

"""
Need to consider the case when too full to have a winning block
"""
import numpy as np

class REFLEX():
    def __init__(self):
        # (x,y) = (horizontal,vertical)
        self.red = (1,1)
        self.blue = (5,5)
        # initial board
        self.board = np.zeros((7, 7))
        self.red_counter = 1
        self.blue_counter = -1
        # winning block with the format of {length:[((start_x,start_y),(end_x,end_y),(1,1,1,1,1)),(...)}
        self.red_winningblock = dict()
        self.blue_winningblock = dict()

    def checkInBetween(self, loc, start, end):

        dx = (end[0]-start[0])//4
        dy = (end[1]-start[1])//4
        step = np.array([0,1,2,3,4])
        x = start[0] + dx*step
        y = start[1] + dy*step

        return loc in list(zip(x.tolist(),y.tolist()))

    def checkOpponent(self, x, y, side):
        # given side, decide if opponent on the way
        if side == 'red':
            return self.board[int(y)][int(x)] < 0
        elif side == 'blue':
            return self.board[int(y)][int(x)] > 0

    def sameWinningBlock(self,start,end,begin,stop):
        return (start == begin and end == stop) or (start == stop and end == begin)

    def findPosition(self,start,end,loc):
        dx = (end[0]-start[0])//4
        dy = (end[1]-start[1])//4
        step = np.array([0,1,2,3,4])
        x = start[0] + dx*step
        y = start[1] + dy*step
        return list(zip(x.tolist(),y.tolist())).index(loc)

    def updateWinningBlock(self, add, current_winningblock, opponent_winningblock,side):
        # update after the current player place the stone and before the next player play
        # update the current player winning block 
        # then update the opponent winning block

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
                    if self.checkOpponent(x_val[i]+increment[idx][0][j], y_val[i]+increment[idx][1][j], side):
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

    def breakATie(self,loc1,loc2):
        if loc1[0]<loc2[0] or (loc1[0]==loc2[0] and loc1[1]<loc2[1]):
            return loc1
        else:
            return loc2

    def checkNeighbour(self,vec):
        """
        set impossible candidate to 1
        """ 
        tmp = np.asarray(vec)
        # left shift
        left_shift = np.roll(tmp,-1)
        left_shift[-1] = 0
        # right shift
        right_shift = np.roll(tmp,1)
        right_shift[0] = 0

        result = tmp+left_shift+right_shift

        return tuple(tmp+(result==0))

    def decodeLocation(self,vec,start,end):
        """
        return a list of coordinates to explore
        """

        dx = (end[0]-start[0])//4
        dy = (end[1]-start[1])//4
        step = np.array([0,1,2,3,4])
        step.astype(int)
        x = start[0] + dx*step
        y = start[1] + dy*step
        idx = np.asarray(vec)
        x = x[idx==0]
        y = y[idx==0]
        return [(x[i],y[i]) for i in range(len(x))]

    def placeStone(self,add,side):
        if side == 'red':
            self.board[add[1]][add[0]] = self.red_counter
            self.red_counter +=1
            print('red stone location'+'('+str(add[0])+','+str(add[1])+')')
        elif side == 'blue':
            self.board[add[1]][add[0]] = self.blue_counter
            self.blue_counter -=1
            print('blue stone location'+'('+str(add[0])+','+str(add[1])+')')

    def secondPriority(self,opponent_winningblock):
        if 4 in opponent_winningblock:
            for i in range(len(opponent_winningblock[4])):
                vec = opponent_winningblock[4][i][2]
                if vec[4] ==0 or vec[0] == 0:
                    return True
        return False

    def thirdPriority(self,opponent_winningblock):
        if 3 in opponent_winningblock:
            for i in range(len(opponent_winningblock[3])):
                vec = opponent_winningblock[3][i][2]
                if vec[0] == 0 and vec[4] == 0:
                    return True

        return False

    def strategy(self, player_winningblock, opponent_winningblock):
        # winning block of the form {length:[((start_x,start_y),(end_x,end_y),(1,1,1,1,1)),(...)}
        """
        Need to implement when there is no winning block
        """
        add = None
        # Check whether the agent side is going to win 
        # by placing just one more stone.
        if 4 in player_winningblock:
            temp = (6,6)
            for i in range(len(player_winningblock[4])):
                candidates = self.decodeLocation(player_winningblock[4][i][2],player_winningblock[4][i][0],player_winningblock[4][i][1])
                temp = self.breakATie(candidates[0],temp)
            add = temp
            return add, False

        elif self.secondPriority(opponent_winningblock):
        # Then check whether the opponent has an UNBROKEN chain formed by 4 stones 
        # and has an empty intersection on either head of the chain. 
            temp = (6,6)
            for i in range(len(opponent_winningblock[4])):
                vec = opponent_winningblock[4][i][2]
                if vec[4] ==0 or vec[0] == 0:
                    candidates = self.decodeLocation(vec,opponent_winningblock[4][i][0],opponent_winningblock[4][i][1])
                    temp = self.breakATie(candidates[0],temp)
            add = temp

        elif self.thirdPriority(opponent_winningblock):
        # Check whether the opponent has an UNBROKEN chain formed by 3 stones and 
        # has empty spaces on both ends of the chain.
            temp = (6,6)
            for i in range(len(opponent_winningblock[3])):
                vec = opponent_winningblock[3][i][2]
                if vec[0] == 0 and vec[4] == 0:
                    candidates = self.decodeLocation(vec,opponent_winningblock[3][i][0],opponent_winningblock[3][i][1])
                    for j in range(2):
                        temp = self.breakATie(candidates[j],temp)
            add = temp

        else:
        # find all possible sequences of 5 consecutive spaces that contain none of the opponent's stones
        # find the winning block which has the largest number of the agent stones
        # Last, in the winning block, place a stone next to a stone already in the winning block on board.
            idx = max(player_winningblock.keys())
            temp = (6,6)
            
            for i in range(len(player_winningblock[idx])):
                valid = self.checkNeighbour(player_winningblock[idx][i][2])
                candidates = self.decodeLocation(valid,player_winningblock[idx][i][0],player_winningblock[idx][i][1])
                for j in range(5-int(sum(valid))):
                    temp = self.breakATie(candidates[j],temp)
            add = temp
        return add, True

    def printResult(self):
        """
        '.' to denote empty intersections
        small letters to denote each of the red stones (first player)
        capital letters to denote each of the blue stones (second player)  

        """
        f = open('2.1result.txt','w')
        for j in reversed(range(7)):
            for i in range (7):
                if self.board[j][i] == 0:
                    f.write('.')
                elif self.board[j][i]>0: 
                    f.write(chr(ord('a')+int(self.board[j][i])-1))
                elif self.board[j][i]<0:
                    f.write(chr(ord('A')+abs(int(self.board[j][i]))-1))
            f.write('\n')
        
        f.close()

        
def main():
    play = True
    # initialize the winning block
    model = REFLEX()
    model.updateWinningBlock(model.red,model.red_winningblock,model.blue_winningblock,'red')
    model.placeStone(model.red,'red')
    model.updateWinningBlock(model.blue,model.blue_winningblock,model.red_winningblock,'blue')
    model.placeStone(model.blue,'blue')

    while play:
        # red first
        add,play = model.strategy(model.red_winningblock,model.blue_winningblock)
        model.updateWinningBlock(add,model.red_winningblock,model.blue_winningblock,'red')
        model.placeStone(add,'red')
        if play == False:
            print('RED WIN')
            break
        # blue second
        add,play = model.strategy(model.blue_winningblock,model.red_winningblock)
        model.updateWinningBlock(add,model.blue_winningblock,model.red_winningblock,'blue')
        model.placeStone(add,'blue')
        if play == False:
            print('BLUE WIN')

    model.printResult()

if __name__ == "__main__":
    main()
