# Part 1.1 T-D learning and SARSA
# fwu11

import numpy as np
from pong_game import GameState
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

def TD_learning(epsilon):
    iteration = 0
    num_bounce = 0
    total_num_bounce = 0
    game_state = GameState()
    Q_table,N_state_action = Q_value()

    f = open("TD.txt",'w')

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
    np.save('TD_Q.npy',Q_table)
    return Q_table

def SARSA(epsilon):
    iteration = 0
    num_bounce = 0
    total_num_bounce = 0
    game_state = GameState()
    Q_table,N_state_action = Q_value()


    f = open("SARSA.txt",'w')

    # initial action: down
    current_action = 2

    while (iteration < 100000):
        #state = (game_state.ball_x,game_state.ball_y,game_state.velocity_x,game_state.velocity_y,game_state.paddle_y)
        #move_gui(state)

        current_state = game_state.discretize_state()
        # update ALPHA
        N_state_action[current_state][current_action]+=1
        ALPHA = C/(C+N_state_action[current_state][current_action])

        # at terminate state
        if( current_state == -1):

            total_num_bounce += num_bounce
            #print("number of bounces "+str(num_bounce))
            num_bounce = 0

            #init_gui_ball(state)

            # update Q value
            Q_table[current_state][current_action] = Q_table[current_state][current_action]+ALPHA*(-1-Q_table[current_state][current_action])

            # restart the game
            game_state = GameState()
            iteration +=1
            f.write(str(total_num_bounce/iteration)+'\n')
            #print("Mean Episode Rewards: " + str(total_num_bounce/iteration))

        else:
            # observe reward R
            current_reward = game_state.reward

            if(current_reward == 1):
                num_bounce+=1

            # next state S'
            game_state.update_state(current_action)
            next_state = game_state.discretize_state()

            # from next state epsilon greedy to choose the next action a'
            next_action = epsilon_greedy(Q_table,epsilon,next_state)
            epsilon = scale_down_epsilon(epsilon)

            # update Q value
            Q_table[current_state][current_action] = Q_table[current_state][current_action]+ALPHA*(current_reward+GAMMA*Q_table[next_state][next_action]-Q_table[current_state][current_action])
            # update the action
            current_action = next_action

    f.close()
    np.save('SARSA_Q.npy',Q_table)
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

    f = open("test.txt",'w')

    while (iteration<200):
        state = (game_state.ball_x,game_state.ball_y,game_state.velocity_x,game_state.velocity_y,game_state.paddle_y)
        move_gui(state)
        current_state = game_state.discretize_state()
        current_reward = game_state.reward
        action_index = np.argmax(Q_table[current_state])
        game_state.update_state(action_index)

        if( current_reward == -1):
            total_num_bounce += num_bounce
            init_gui_ball(state)
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
    switch = "TD_learning"
    #switch = "SARSA"

    epsilon = INITIAL_EPSILON

    #init_gui()
    '''
    # training
    print("start training with "+switch)
    time1 = time.time()
    if switch == "TD_learning":
        Q_table = TD_learning(epsilon)
    else:
        Q_table = SARSA(epsilon)
    time2 = time.time()

    print("training time "+str(time2-time1))
    # setup the game environment
    print("creating gui")
    #init_gui()
    '''
    # Load
    ### Q does not store/load correctly
    Q_table = np.load('TD_Q.npy').item()
    #print(Q_table)
    # testing
    init_gui()
    print("start testing")
    testing(Q_table)


if __name__ == "__main__":
    main()
