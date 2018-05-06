# Part 1.1 T-D learning and SARSA
# fwu11

import numpy as np
from part2extra_pong_game import GameState
from DeepNeuralNetwork import DNN


def Q_value():
    Q_table = dict()
    N_state_action = dict()
    for s0 in range(12):  # ball position x
        for s1 in range(12):  # ball position y
            for s2 in [-1, 1]:  # velocity in x direction
                for s3 in [-1, 0, 1]:  # velocity in y direction
                    for s4 in range(12):  # position of paddle
                        # state = (s0,s1,s2,s3,s4)
                        # 3 actions: 0 stay, 1 paddle up, 2 paddle down
                        Q_table[(s0, s1, s2, s3, s4)] = np.zeros(3)
                        N_state_action[(s0, s1, s2, s3, s4)] = np.zeros(3)

    # add a terminal state with key of -1
    Q_table[-1] = np.zeros(3)
    N_state_action[-1] = np.zeros(3)
    return Q_table,N_state_action


def testing(Q_table,dnn):
    iteration = 0
    game_state = GameState()

    f = open("extracredit.txt",'w')

    while (iteration<1000):
        current_state = game_state.return_state()
        current_state_d = game_state.discretize_state()
        action_1_index = np.argmax(Q_table[current_state_d])
        action_2_index = dnn.predict(current_state)
        game_state.update_state(action_1_index,action_2_index)

        if( game_state.terminate == True):
            f.write(str(game_state.bounce_1)+':'+ str(game_state.bounce_2)+'\n')
            game_state = GameState()
            iteration +=1
    f.close()

def read_data(filename):
	data, label = [], []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line_split = line.strip().split(' ')
			data.append([float(num) for num in line_split[:-1]])
			label.append(int(float(line_split[-1])))
	return data, label

def train_test_split(data, label, split=0.2):
	N = len(data)
	limit = int(N * (1 - split))
	return data[:limit], label[:limit], data[limit:], label[limit:]

def label_to_one_hot(label):
	Y = [[0]*len(set(label)) for _ in range(len(label))]
	for i, num in enumerate(label):
		Y[i][num] = 1
	return Y


def main():

    filename = "expert_policy.txt"
    data, label = read_data(filename)
    Y = label_to_one_hot(label)
    data = np.array(data)
    Y = np.array(Y)

    X_train, y_train, X_test, y_test = train_test_split(data, Y)

    dnn = DNN()
    dnn.fit(X_train, y_train)

    Q_table = np.load('part2_agentB_Q.npy').item()

    # testing

    print("start testing")
    testing(Q_table,dnn)


if __name__ == "__main__":
    main()