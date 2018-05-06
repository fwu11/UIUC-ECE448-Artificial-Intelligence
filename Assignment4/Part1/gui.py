from tkinter import *
from random import randint

class Pad:
    def __init__(self, canvas_input, pad_position, pad_x = 500, pad_width = 6.5, pad_height = 100):
        # position of the pad
        self.pad_y = pad_position
        self.canvas = canvas_input
        # initial position of the pad
        pad_left = pad_x - pad_width
        points = [pad_x, pad_position+pad_height, pad_left, pad_position+pad_height, pad_left, pad_position, pad_x, pad_position]
        self.pad = canvas_input.create_polygon(points, fill='black')

    def move_pad(self, delta_y):
        # update the pad position
        self.canvas.move(self.pad, 0, delta_y)
        self.pad_y += delta_y
    

class Ball:
    def __init__(self, canvas_input, x, y):
        self.x = x
        self.y = y
        self.canvas = canvas_input
        self.ball = canvas_input.create_oval(self.x, self.y, self.x+10, self.y+10, fill="red")

    def move_ball(self, delta_x, delta_y):
    	# move ball
        self.canvas.move(self.ball, delta_x, delta_y)
        # update position
        self.x += delta_x
        self.y += delta_y

canvas = None
ball = None
pad = None
root = None
# initialize Window and canvas, and create two ball objects and animate them
def init_gui():
    global canvas
    global ball
    global pad
    global root
    root = Tk()
    root.title("Part1.1")
    canvas = Canvas(root, width = 500, height = 500)
    if canvas == None:
        print("error!")
    canvas.pack()
    pad = Pad(canvas, 0.4 * 500)
    ball = Ball(canvas, 250, 250)
    

def init_gui_ball(game_status):
    global canvas
    global ball
    global pad
    ball.move_ball(ball.x - game_status[0] * 500, ball.y - game_status[1] * 500)
    canvas.update()


def move_gui(game_status):
    global canvas
    global ball
    global pad

    # move paddle
    delta_pad_y = (500 * game_status[4])-pad.pad_y
    pad.move_pad(delta_pad_y)

    #move ball
    delta_x = 500*game_status[0] - ball.x
    delta_y = 500*game_status[1] - ball.y
    ball.move_ball(delta_x, delta_y)

    canvas.after(50)
    canvas.update()
    return 0