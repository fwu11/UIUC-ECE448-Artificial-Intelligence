# implement Assignment 1 part 1.2
import numpy as np
from multipleSearch import MultipleSearch
import time
import heapq

def main():
    start_time = time.time()
    filename = 'small'
    Maze, start, dst,dst_order = readFile('data/' + filename + 'Search.txt')
    model = MultipleSearch(Maze,start,dst,dst_order)
    model.search(f)
    model.plot(filename)
    print(time.time()-start_time)


def readFile(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    height = len(lines)
    dst = {}
    dst_order =[]
    counter = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if i == 0:
            width = len(line)
            Maze = -np.ones((height, width))
        for j, element in enumerate(line):
            if element == '%':
                Maze[i][j] = -2
            if element == '.':
                dst[(i, j)] = counter
                dst_order.append((i,j))
                counter +=1
            if element == 'P':
                start = (i, j)
    f.close()
    return Maze, start, dst,dst_order


def f(queue):
    popped_pos = heapq.heappop(queue)

    return popped_pos, queue


if __name__ == "__main__":
    main()
