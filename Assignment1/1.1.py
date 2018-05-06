#implement Assignment 1 part 1.1
from search import Search
import numpy as np
import os


def main():
    filename = 'open'
    searchMethod = 'gbfs'

    Maze,start,dst = readFile('data/'+ filename+'Maze.txt')
    model = Search(Maze,start,dst)
    if searchMethod == 'bfs':
        model.search(h_bfs,None)
    elif searchMethod == 'dfs':
        model.search(h_dfs,None)
    elif searchMethod == 'gbfs':
        model.search(h_gbfs,f)
    elif searchMethod == 'astar':
        model.search(h_astar,f)

    model.plot(filename,searchMethod)

def readFile(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    height = len(lines)
    for i, line in enumerate(lines):
        line = line.strip()
        if i == 0:
            width = len(line)
            Maze = np.zeros((height, width))
        for j, element in enumerate(line):
            if element == '%':
                Maze[i][j] = 1
            if element =='.':
                dst = (i,j)
            if element =='P':
                start = (i,j)
    f.close()
    return Maze, start, dst    

def h_bfs(l):
    return l[0],l[1:]

def h_dfs(l):
    return l[-1],l[:-1]

def h_gbfs(x,y,dst,path_cost):
    return abs(x-dst[0])+abs(y-dst[1])


def h_astar(x,y,dst,path_cost):
    return abs(x-dst[0])+abs(y-dst[1])+path_cost[(x,y)]

def f(h,path_cost,l,dst):
    minimum = float("infinity")
    loc = 0
    for i in range(len(l)):
        x,y = l[i]
        heuristic = h(x,y,dst,path_cost)
        if heuristic < minimum:
            loc = i
            minimum = heuristic

    popped_pos = l.pop(loc)
    queue = l
    return popped_pos, queue

if __name__ == "__main__":
    main()

