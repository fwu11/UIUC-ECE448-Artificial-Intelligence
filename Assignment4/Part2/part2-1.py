import numpy as np
from DeepNeuralNetwork import DNN
from pong_game import GameState

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


def pong_test(model):
	iteration = 0
	num_bounce = 0
	total_num_bounce = 0
	max_num_bounce = 0
	game_state = GameState()

	while iteration < 200:
		state = np.array([game_state.ball_x,game_state.ball_y,game_state.velocity_x,game_state.velocity_y,game_state.paddle_y])
		current_reward = game_state.reward
		action_index = model.predict(state)
		game_state.update_state(action_index)

		if  game_state.terminate == True:
			total_num_bounce += num_bounce
			max_num_bounce = max(num_bounce, max_num_bounce)
			num_bounce = 0
			game_state = GameState()
			iteration +=1
		else:
			if current_reward == 1:
				num_bounce += 1
	print("max number of bounces "+str(max_num_bounce))
	print("Average number of bounces in testing: "+str(total_num_bounce/200)+'\n')

def main():

	filename = "expert_policy.txt"
	data, label = read_data(filename)
	Y = label_to_one_hot(label)
	data = np.array(data)
	Y = np.array(Y)

	dnn = DNN(epochs=500)
	dnn.fit(data, Y)

	print('Start testing pong game...')
	pong_test(dnn)

if __name__ == "__main__":
    main()


