# Part2.1
# fwu11

import numpy as np 

class affine_cache:
    def __init__(self,A,W,b):
        self.A = A
        self.W = W
        self.b = b

class relu_cache:
    def __init__(self,Z):
        self.Z = Z

#The actions 0 represents the paddle moving up, 1 represents the paddle maintaining its position, and 2 represents the paddle moving down

def read_data(data_txt_file):
    f = open(data_txt_file,'r')
    lines = f.readlines()
    N = len(lines)
    data = np.zeros((N,5))
    label = np.zeros(N)
    counter = 0
    
    for line in lines:
        line = line.rstrip('\n')
        data[counter][0] = float(line.split(' ')[0])
        data[counter][1] = float(line.split(' ')[1])
        data[counter][2] = float(line.split(' ')[2])
        data[counter][3] = float(line.split(' ')[3])
        data[counter][4] = float(line.split(' ')[4])
        label[counter] = float(line.split(' ')[5])
        counter +=1
    f.close()

    return data, label

def affine_forward(A,W,b):

    Z = np.dot(A,W)+b
    acache = affine_cache(A,W,b)

    return Z,acache

def affine_backward(dZ,acache):
    A = acache.A 
    W = acache.W

    dA = np.dot(dZ,W.T)
    dW = np.dot(A.T,dZ)
    db = np.sum(dZ,axis=0)
    
    return dA,dW,db

def relu_forward(Z):

    A = np.maximum(0,Z)
    rcache = relu_cache(Z)

    return A,rcache

def relu_backward(dA,rcache):
    Z = rcache.Z
    dZ = dA*(Z>0.0)

    return dZ 

def cross_entropy(F,y):
    N = len(y)

    L =  -1/N * np.sum(np.choose(y,F.T)-np.log(np.sum(np.exp(F),axis = 1)))

    dF = -1/N *(1.0*(np.repeat(np.array([[0,1,2]]),N,axis = 0)==y)-np.exp(F)/(np.sum(np.exp(F),axis = 1))[:,None])

    return L,dF


def NN(X,y,test,learning_rate,W1,W2,W3,W4,b1,b2,b3,b4):

    # forward
    Z1,acache1 = affine_forward(X,W1,b1)
    A1,rcache1 = relu_forward(Z1)
    Z2,acache2 = affine_forward(A1,W2,b2)
    A2,rcache2 = relu_forward(Z2)
    Z3,acache3 = affine_forward(A2,W3,b3)
    A3,rcache3 = relu_forward(Z3)
    F,acache4 = affine_forward(A3,W4,b4)

    loss,dF = cross_entropy(F,y)

    # backward
    dA3,dW4,db4 = affine_backward(dF,acache4)
    dZ3 = relu_backward(dA3,rcache3)
    dA2,dW3,db3 = affine_backward(dZ3,acache3)
    dZ2 = relu_backward(dA2,rcache2)
    dA1,dW2,db2 = affine_backward(dZ2,acache2)
    dZ1 = relu_backward(dA1,rcache1)
    dX,dW1,db1 = affine_backward(dZ1,acache1)

    # update parameters
    W1 = W1-learning_rate*dW1
    W2 = W2-learning_rate*dW2
    W3 = W3-learning_rate*dW3
    W4 = W4-learning_rate*dW4
    b1 = b1-learning_rate*db1
    b2 = b2-learning_rate*db2
    b3 = b3-learning_rate*db3
    b4 = b4-learning_rate*db4

    return loss

class weight:
    def __init__(self,dim1,dim2):
        pass
        
class bias:
    def __init__(self,size):
        self.b = np.zeros(size)



def minibatchGD(data,label,epoch,batch_size,learning_rate):
    N = len(label)
    # initialize parameters
    W1
    W2
    W3
    W4

    b1 = np.zeros(256)
    b2 = np.zeros(256)
    b3 = np.zeros(256)
    b4 = np.zeros(3)

    for e in range(epoch):
        # shuffle data
        np.random.shuffle()
        for i in range(int(np.ceil(N/batch_size))):
            X = 
            y = 
            loss = NN(X,y,False,learning_rate,W1,W2,W3,W4,b1,b2,b3,b4)


def main():
    data_txt_file = "expert_policy.txt"
    data, label = read_data(data_txt_file)
    label = label.astype(int)
    batch_size = 128

if __name__ == "__main__":
    main()