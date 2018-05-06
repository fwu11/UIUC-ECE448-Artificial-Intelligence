import heapq

class linkedList():
    def __init__(self, value):
        # state consists of ((loc),(dst_array))
        self.state = value
        self.parent = None


list_head = linkedList(((1,2),(1,1,1)))
queue = []
heapq.heappush(queue,(5,list_head,0))
node2 = linkedList(((1,3),(1,1,0)))
heapq.heappush(queue,(6,node2,0))
heapq.heappush(queue,(6,node2,0))
print(queue)
node = heapq.heappop(queue)
print(queue)


print(node2.parent)
