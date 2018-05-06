import numpy as np
from DeepNeuralNetwork import DNN

def read_data(filename):
	data, label = [], []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line_split = line.strip().split(' ')
			data.append([float(num) for num in line_split[:-1]])
			label.append(int(float(line_split[-1])))
	return data, label

def train_test_split(data, label, split=0.3):
	N = len(data)
	limit = int(N * (1 - split))
	return data[:limit], label[:limit], data[limit:], label[limit:]

def label_to_one_hot(label):
	Y = [[0]*len(set(label)) for _ in range(len(label))]
	for i, num in enumerate(label):
		Y[i][num] = 1
	return Y

filename = "expert_policy.txt"
data, label = read_data(filename)
Y = label_to_one_hot(label)
data = np.array(data)
Y = np.array(Y)

#X_train, y_train, X_test, y_test = train_test_split(data, Y)

dnn = DNN()
dnn.fit(data, Y)
pred = dnn.predict(data)
pr, re = np.argmax(pred, axis=1), np.argmax(Y, axis=1)
confusion_matrix = np.zeros((3,3))
for i in range(len(pr)):
	confusion_matrix[re[i]][pr[i]] += 1
for line in confusion_matrix:
	print (line)
print('The test accuracy is ' + str(float(confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2])/float(len(pr))))
