import numpy as np

def read_dataset(data_txt_file):
    data = dict()
    f = open(data_txt_file, 'r')
    rows = f.readlines()
    N = len(rows)//33
    image = np.zeros((N,32,32))
    label = np.zeros(N)
    counter = 1
    for line in rows:
        if counter %33 == 0:
            label[counter//33 -1] = int(line.strip())
        else:
            image[counter//33][counter%33 -1] = np.fromstring(line.rstrip('\n'),np.int8)-48
        counter += 1      
    f.close()
    data = {'image':image,'label':label}
    return data

def read_facedata(data_txt_file):
    f = open(data_txt_file, 'r')
    rows = f.readlines()
    N = len(rows)//70
    image = np.zeros((N,70,60))
    counter = 0
    for line in rows:
        line = line.replace('#', '1')
        line = line.replace(' ','0')
        image[counter//70][counter%70] = np.fromstring(line.rstrip('\n'),np.int8)-48     
        counter+=1
    f.close()
    return image

def read_facelabel(label_txt_file):
    f = open(label_txt_file, 'r')
    rows = f.readlines()
    N = len(rows)
    label = np.zeros(N)
    counter = 0
    for line in rows:
        label[counter] = int(line.strip())
        counter +=1
    f.close()

    return label
