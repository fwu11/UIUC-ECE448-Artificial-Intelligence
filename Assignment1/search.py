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
        self.path_cost = {start:0}
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

    def validNeighbour(self,x_o,y_o,x,y,visited,path_cost):
        # out of bound
        if x<0 or x > self.row-1:
            return False
        if y<0 or y > self.col-1:
            return False
        # Wall
        if self.Maze[x,y] == 1:
            return False
        # Visited already
        if (x,y) in visited and path_cost[(x,y)] <= path_cost[(x_o,y_o)]+1:
            return False
        return True

    def search(self,h,f):
        queue = [self.start]
        visited = set(self.start)
        stop = True

        while stop:
            #bfs,dfs
            if f is None:
                popped_pos, queue = h(queue)
            else:
                #gbfs,astar
                popped_pos, queue = f(h,self.path_cost,queue,self.dst)

            x,y = popped_pos
            self.node_expand +=1

            # check each of the four neighbours
            x_shift = [0, 1, 0, -1]
            y_shift = [1, 0, -1, 0]
            for i in range(4):
                if self.validNeighbour(x,y,x_shift[i]+x,y_shift[i]+y,visited,self.path_cost):
                    node = linkedList((x_shift[i]+x,y_shift[i]+y))
                    node.parent = self.map[(x,y)]
                    self.map[(x_shift[i]+x,y_shift[i]+y)]= node
                    if not (x_shift[i]+x,y_shift[i]+y) in self.path_cost or ((x_shift[i]+x,y_shift[i]+y) in self.path_cost and self.path_cost[(x_shift[i]+x,y_shift[i]+y)] > self.path_cost[(x,y)]+1):
                        self.path_cost[(x_shift[i]+x,y_shift[i]+y)] = self.path_cost[(x,y)]+1

                    queue.append((x_shift[i]+x,y_shift[i]+y))
                    visited.add((x_shift[i]+x,y_shift[i]+y))

                    if node.val == self.dst:
                        self.cost = self.path_cost[(x_shift[i]+x,y_shift[i]+y)]
                        print('finished')
                        print('path cost: '+str(self.cost))
                        print('node expanded: '+str(self.node_expand))
                        stop = False
