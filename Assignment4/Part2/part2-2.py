import numpy as np
from QLearningNetwork import QLearning
from pong_game import GameState

def read_data(filename):
	data, label = [], []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line_split = line.strip().split(' ')
			data.append([float(num) for num in line_split[:-3]])
			label.append([float(num) for num in line_split[-3:]])
	return data, label

def train_test_split(data, label, split=0.2):
	N = len(data)
	limit = int(N * (1 - split))
	return data[:limit], label[:limit], data[limit:], label[limit:]

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
	filename = "advantages.txt"
	data, label = read_data(filename)
	data = np.array(data)
	label = np.array(label)

	learning_rate_list = [0.001, 0.01, 0.1]
	batch_size_list = [8, 128, 1024]
	num_units_list = [16, 64, 256]

	# Tune the hyperparameter of neural network
	for learning_rate in learning_rate_list:
		for batch_size in batch_size_list:
			for num_units in num_units_list:
				dnn = QLearning(learning_rate=learning_rate, batch_size=batch_size, num_units=num_units)
				dnn.fit(data, label)
				acc = dnn.test(data, label)
				print('Learning rate: ' + str(learning_rate) + ', batch size: ' + str(batch_size) + ', number of units: ' + str(num_units) + ', train loss is ' + str(acc))

	# The best choice of hyperparameters combination is
	# learning rate: 0.01
	# batch size: 8
	# number of units per layer: 256
	dnn = QLearning(learning_rate=0.01, batch_size=8, num_units=256, plot_loss=True)
	dnn.fit(data, label)
	#acc = dnn.test(data, label)

	# Testing using Pong environment
	print("")
	print('Start testing pong game...')
	pong_test(dnn)

if __name__ == "__main__":
    main()
