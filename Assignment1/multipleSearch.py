# 1.2
import numpy as np
import copy
import heapq

class linkedList():
    def __init__(self, value):
        # state consists of ((loc),(dst_array))
        self.state = value
        self.parent = None
    def __lt__(self,other):
        return sum(self.state[1]) <= sum(other.state[1])

class MultipleSearch():
    def __init__(self, Maze, start, dst, dst_order):
        self.Maze = Maze
        self.start = start
        # dst dictionary, element unchanged
        self.dst = dst
        self.node_expand = 0
        self.final_dst = None
        self.list_head = None
        self.num_dst = len(dst)
        self.dst_order = np.array(dst_order)
        # in the form of (1,1,1,1,...,1) each element represent one dot and use dst to lookup the coordinates
        self.dst_array = tuple([1 for x in range(len(dst))])
        #self.map = {(start,self.dst_array): linkedList((start,self.dst_array))}
        self.cost = 0
        self.row = Maze.shape[0]
        self.col = Maze.shape[1]
        self.goalSequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e',
                             'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                             't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                             'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                             'V', 'W', 'X', 'Y', 'Z']
        self.matrix = None

    def plot(self,filename):
        '''
        plot the Maze
        '''
        counter = self.num_dst
        current = self.final_dst

        #while current.state != (self.start,tuple([1 for x in range(self.num_dst)])):
        while current != self.list_head:
            prev = current.parent
            if current.state[1] != prev.state[1] :
                counter -= 1
                self.Maze[current.state[0][0]][current.state[0][1]] = counter

            current = current.parent

        f = open(filename+' result.txt', 'w')
        for i in range(self.row):
            for j in range(self.col):
                if (i, j) == self.start:
                    f.write('P')
                elif self.Maze[i][j] == -2:
                    f.write('%')
                elif self.Maze[i][j] == -1:
                    f.write(' ')
                else:
                    goal = int(self.Maze[i][j])
                    f.write(str(self.goalSequence[goal]))
            f.write('\n')
        f.write('Path Cost: '+ str(self.cost)+'\n')
        f.write('Node Expanded: '+ str(self.node_expand))
        f.close()

    def validNeighbour(self, x, y, dst_remain, visited):
        '''
        Check if the next state is a valid state to explore
        '''
        # out of bound
        if x < 0 or x > self.row - 1:
            return False
        if y < 0 or y > self.col - 1:
            return False
        # Wall
        if self.Maze[x][y] == -2:
            return False
        # State visited already
        if ((x, y), dst_remain) in visited:
            return False
        return True

    def removeDot(self,dst_array,dot,dst):
        '''
        When the dot is visited, the dst_array element is set to 0
        e.g.(1,1,1,0,1,...1)
        '''
        temp = list(dst_array)
        index = dst[dot]
        temp[index] = 0
        return tuple(temp)

    def preComputeHeuristic(self,dst_order):
        n = len(dst_order)
        matrix = np.zeros((n,n))
        for i in range(n):
            a,b = dst_order[i]
            for j in range(i+1, n):
                c,d = dst_order[j]
                matrix[i][j] = abs(a-c)+abs(b-d)

        return matrix


    def computeHeuristic(self, loc, dst_array, dst_order,matrix):
        x,y = loc
        n = sum(dst_array)
        dst_array = np.array(dst_array)
        dst = dst_order[dst_array==1]
        temp = matrix[dst_array ==1,:]
        temp = temp[:,dst_array==1]
        h = 0

        if n==1:
            h = abs(x-dst[0][0])+abs(y-dst[0][1])
        elif n==0:
            h =0
        else:
            h = np.max(temp)
            l = np.argmax(temp)
            a, b = dst[l//n]
            c, d = dst[l%n]
            '''
            for i in range(n):
                a, b = dst[i]
                #dist = abs(a - x)+abs(b - y)
                #if(dist < h):
                #    h = dist
                for j in range(i + 1, n):
                    c, d = dst[j]
                    dist = abs(a - c)+abs(b - d)
                    if(dist > h):
                        h = dist
                        one = (a,b)
                        two = (c,d)
            '''
            h += min(abs(x - a)+abs(y - b), abs(x - c)+abs(y - d))

        return h

    def search(self, f):
        '''
        A* search
        '''
        self.matrix = self.preComputeHeuristic(self.dst_order)

        # elements in the queue are the states
        h = self.computeHeuristic(self.start,self.dst_array,self.dst_order,self.matrix)
        
        # queue of the format(path_cost+heuristic,loc,dst_array,path_cost)
        # use priority queue implementation
        self.list_head = linkedList((self.start,self.dst_array))
        queue = []
        heapq.heappush(queue,(h,self.list_head,0))
        
        visited = set()
        visited.add((self.start, self.dst_array))
        stop = True

        

        while stop:
            # expand node
            popped, queue = f(queue)
            x, y = popped[1].state[0]
            dst_remain = popped[1].state[1]
            path_cost = popped[2]
            self.node_expand +=1


            # check each of the four neighbours
            x_shift = [0, 1, 0, -1]
            y_shift = [1, 0, -1, 0]
            for i in range(4):
                if self.validNeighbour(x_shift[i] + x, y_shift[i] + y, dst_remain, visited):
                    dst_from_parent = copy.deepcopy(dst_remain)
                    if (x_shift[i] + x, y_shift[i] + y) in self.dst and dst_from_parent[self.dst[(x_shift[i] + x, y_shift[i] + y)]]==1:
                        dst_current = self.removeDot(dst_from_parent,(x_shift[i] + x, y_shift[i] + y),self.dst)
                    else:
                        dst_current = dst_from_parent

                    visited.add(((x_shift[i] + x, y_shift[i] + y), dst_current))
                    node = linkedList(((x_shift[i] + x, y_shift[i] + y), dst_current))
                    node.parent = popped[1]
                    h = self.computeHeuristic((x_shift[i] + x, y_shift[i] + y),dst_current,self.dst_order,self.matrix)
                    heapq.heappush(queue,(path_cost+1+h,node,path_cost+1))

                    
                    #node.parent = self.map[((x, y),dst_from_parent)]
                    #self.map[((x_shift[i] + x, y_shift[i] + y),dst_current)] = node

                    if sum(dst_current) == 0:
                        self.cost = path_cost+1
                        print('finished' + '(' + str(x_shift[i] + x) + ',' + str(y_shift[i] + y) + ')')
                        print('path cost: '+str(path_cost+1))
                        print('node expanded: '+str(self.node_expand))
                        #self.final_dst = (x_shift[i] + x, y_shift[i] + y)
                        self.final_dst = node
                        stop = False
