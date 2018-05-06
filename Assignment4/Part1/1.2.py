# Part1.2
# fwu11


import numpy as np
from part2_pong_game import GameState
from gui import *
import time

# Learning rate constant
C = 10

# Discount factor
GAMMA = 0.9

# initial epsilon greedy parameter
# use decaying epsilon strategy
# Final value of epsilon.
FINAL_EPSILON = 0.05

# Starting value of epsilon.
INITIAL_EPSILON = 1.0

# Frames over which to anneal epsilon.
EXPLORE = 80000

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

def scale_down_epsilon(epsilon):
    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    return epsilon

def TD_learning(epsilon,Q_table,N_state_action):
    iteration = 0
    num_bounce = 0
    total_num_bounce = 0
    game_state = GameState()

    f = open("part2_agentB.txt",'w')

    while (iteration < 100000):
        #state = (game_state.ball_x,game_state.ball_y,game_state.velocity_x,game_state.velocity_y,game_state.paddle_y)
        #move_gui(state)

        # epsilon greedy to choose the action
        current_state = game_state.discretize_state()
        action_index = epsilon_greedy(Q_table,epsilon,current_state)
        epsilon = scale_down_epsilon(epsilon)

        # update ALPHA
        N_state_action[current_state][action_index]+=1
        ALPHA = C/(C+N_state_action[current_state][action_index])

        # observe reward R
        current_reward = game_state.reward

        # at terminate state
        if( current_state == -1):
            total_num_bounce += num_bounce
            #print("number of bounces "+str(num_bounce))
            num_bounce = 0

            #init_gui_ball(state)

            # update Q value
            Q_table[current_state][action_index] = Q_table[current_state][action_index]+ALPHA*(-1-Q_table[current_state][action_index])

            # restart the game
            game_state = GameState()
            iteration +=1
            f.write(str(total_num_bounce/iteration)+'\n')
        else:
            # next state S'
            game_state.update_state(action_index)
            next_state = game_state.discretize_state()

            if(current_reward == 1):
                num_bounce+=1

            # update Q value
            Q_table[current_state][action_index] = Q_table[current_state][action_index]+ALPHA*(current_reward+GAMMA*np.max(Q_table[next_state])-Q_table[current_state][action_index])
    f.close()
    np.save('part2_agentB_Q.npy',Q_table)
    return Q_table


def epsilon_greedy(Q_table,epsilon,game_state):
    # perform epislon greedy to balance exploration and exploitation
    # return the action 

    if np.random.uniform() < epsilon:
        action_index = np.random.randint(3)
    else:
        action_index = np.argmax(Q_table[game_state])

    return action_index


def testing(Q_table):
    iteration = 0
    num_bounce = 0
    max_num_bounce = 0
    total_num_bounce = 0
    game_state = GameState()

    f = open("part2_agentB_test.txt",'w')

    while (iteration<200):
        #state = (game_state.ball_x,game_state.ball_y,game_state.velocity_x,game_state.velocity_y,game_state.paddle_y)
        #move_gui(state)
        current_state = game_state.discretize_state()
        current_reward = game_state.reward
        action_index = np.argmax(Q_table[current_state])
        game_state.update_state(action_index)

        if( current_reward == -1):
            total_num_bounce += num_bounce
            #init_gui_ball(state)
            #print("number of bounces "+str(num_bounce))
            f.write(str(num_bounce)+'\n')
            max_num_bounce = max(num_bounce, max_num_bounce)
            num_bounce = 0
            game_state = GameState()
            iteration +=1
        else:
            if (current_reward == 1):
                num_bounce += 1
    print("max number of bounces "+str(max_num_bounce))
    print(str(total_num_bounce/200)+'\n')
    
    f.close()

def main():
    agent = "A"
    #agent = "B"
    epsilon = INITIAL_EPSILON

    #init_gui()

    # training
    print("start training")
    time1 = time.time()

    if agent == "A":
        # load from part 1.1
        _,N_state_action = Q_value()
        Q_table = np.load('TD_Q.npy').item()
        print(Q_table)
        Q_table = TD_learning(epsilon,Q_table,N_state_action)
        print(Q_table)
    else:
        Q_table,N_state_action = Q_value()
        Q_table = TD_learning(epsilon,Q_table,N_state_action)
    time2 = time.time()

    print("training time "+str(time2-time1))
    # setup the game environment
    print("creating gui")
    #init_gui()

    # Load
    #Q_table = np.load('TD_Q.npy').item()
    #init_gui()
    # testing
    print("start testing")
    testing(Q_table)


if __name__ == "__main__":
    main()