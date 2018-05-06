# fwu11

import numpy as np
import math

class GameState:
    def __init__(self):
        self.paddle_height = 0.2
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.velocity_x = 0.03
        self.velocity_y = 0.01
        self.paddle_y = 0.5 - self.paddle_height // 2
        self.reward = 0
        self.terminate = False 

    def update_state(self,action_index):

        # Update the paddle position
        if action_index == 1:
            self.paddle_y -= 0.04
        elif action_index == 2:
            self.paddle_y += 0.04

        # Bounds of movement.
        if self.paddle_y > 1 - self.paddle_height:
            self.paddle_y = 1 - self.paddle_height
        elif self.paddle_y < 0:
            self.paddle_y = 0

        # Update the position of the ball.
        old_ball_x = self.ball_x
        old_ball_y = self.ball_y
        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y

        # Collisions on sides.
        if self.ball_y < 0:
            self.ball_y = -self.ball_y
            self.velocity_y = -self.velocity_y
        
        if self.ball_y > 1:
            self.ball_y = 2 - self.ball_y
            self.velocity_y = -self.velocity_y

        if self.ball_x < 0:
            self.ball_x = -self.ball_x
            self.velocity_x = -self.velocity_x

        # The ball hitting paddle.
        if self.ball_x >= 1:
            slope = (old_ball_y - self.ball_y)/(old_ball_x - self.ball_x)
            intercept = slope*(1 - old_ball_x) + old_ball_y
            if intercept >= self.paddle_y and intercept <= self.paddle_y+self.paddle_height:
                U = np.random.uniform(-0.015, 0.015)
                while(abs(-self.velocity_x + U) <= 0.03 or abs(-self.velocity_x + U) >=1):
                    U = np.random.uniform(-0.015, 0.015)
                self.velocity_x = -self.velocity_x + U
                V = np.random.uniform(-0.03, 0.03)
                while(abs(self.velocity_y + V) >=1):
                    V = np.random.uniform(-0.03, 0.03)
                self.velocity_y = self.velocity_y + V
                self.ball_x = 2 - self.ball_x
                self.reward = 1 
            else:
                self.reward =-1 
                self.terminate = True
        else:
            self.reward = 0 

    def discretize_state(self):

        if self.terminate == True:
            output_state = -1
        else:
            # discretize the board
            if(self.ball_x == 1):
                state_ball_x = 11
            else:
                state_ball_x = int(math.floor(12*self.ball_x))

            if(self.ball_y == 1):
                state_ball_y = 11
            else:
                state_ball_y = int(math.floor(12*self.ball_y))

            # discretize x-velocity
            if(self.velocity_x > 0):
                state_velocity_x = 1
            else:
                state_velocity_x = -1

            # discretize y-velocity
            if(self.velocity_y > 0.015):
                state_velocity_y = 1
            elif(abs(self.velocity_y) <= 0.015):
                state_velocity_y = 0
            else:
                state_velocity_y = -1

            # discretize the paddle
            if(self.paddle_y == 0.8):
                state_paddle_y = 11
            else:
                state_paddle_y = int(math.floor(12 * self.paddle_y / 0.8))

            output_state = (state_ball_x, state_ball_y, state_velocity_x, state_velocity_y, state_paddle_y)
            
        return output_state