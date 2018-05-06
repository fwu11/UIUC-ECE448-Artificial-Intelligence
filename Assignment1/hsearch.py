# 1.1
import numpy as np 

class linkedList():
    def __init__(self,value):
        self.val = value
        self.parent = None


class Search():
    def __init__(self,Maze,start,dst):
        self.Maze = Maze
        self.start = start
        self.dst = dst
        self.map = {start:linkedList(start)}
        #self.path_cost = {start:0}
        self.row = Maze.shape[0]
        self.col = Maze.shape[1]
        self.node_expand = 0
        self.cost = 0

    def plot(self,filename,searchMethod):
        current = self.map[self.dst]
        while current.val != self.start:
            self.Maze[current.val[0],current.val[1]] = 6
            current = current.parent

        f = open(filename+'_'+searchMethod+'.txt','w')
        for i in range(self.row):
            for j in range(self.col):
                if (i,j) == self.dst:
                    f.write('*')
                elif (i,j) == self.start:
                    f.write('P')
                elif self.Maze[i][j] == 1:
                    f.write('%')
                elif self.Maze[i][j] == 0:
                    f.write(' ')
                elif self.Maze[i][j] == 6:
                    f.write('.')
            f.write('\n')
        f.write('Path Cost: '+ str(self.cost)+'\n')
        f.write('Node Expanded: '+ str(self.node_expand))
        f.close()

    def validNeighbour(self,x,y,visited):
        # out of bound
        if x<0 or x > self.row-1:
            return False
        if y<0 or y > self.col-1:
            return False
        # Wall
        if self.Maze[x,y] == 1:
            return False
        # Visited already
        if (x,y) in visited:
            return False
        return True

    def search(self,h,f):
        queue = [(0,self.start)]
        #visited = set()
        visited = set(self.start)
        stop = True
    
        while stop:
            #bfs,dfs
            if f is None:
                popped_pos, queue = h(queue)
            else:
                #gbfs,astar
                popped_pos, queue = f(h,queue,self.dst)

            x,y = popped_pos[1]
            #visited.add((x,y))
            path_cost = popped_pos[0]
            self.node_expand +=1

            # check each of the four neighbours
            x_shift = [-1, 1, 0, 0]
            y_shift = [0, 0, -1, 1]
            for i in range(4):
                if self.validNeighbour(x_shift[i]+x,y_shift[i]+y,visited):
                    node = linkedList((x_shift[i]+x,y_shift[i]+y))
                    node.parent = self.map[(x,y)]
                    self.map[(x_shift[i]+x,y_shift[i]+y)]= node
                    queue.append((path_cost+1, (x_shift[i]+x,y_shift[i]+y)))
                    visited.add((x_shift[i]+x,y_shift[i]+y))

                    if node.val == self.dst:
                        self.cost = path_cost+1
                        print('finished')
                        print('path cost: '+str(self.cost))
                        print('node expanded: '+str(self.node_expand))
                        stop = False
