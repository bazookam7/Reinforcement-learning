import pygame
from random import randint
# from dqn_pacman_priority import DQNAgent
from dqn_pacman import DQNAgent
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set options to activate or deactivate the game view, and its speed
display_option = True
speed = 0
pygame.font.init()
W = 0
H = 0

class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SnakeGen')
        global W,H
        W = 40
        H = 50
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height+60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.wall = Walls()
        self.enemy = Enemy()
        self.score = 0


class Player(object):

    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/player.png')
        self.x_change = 20
        self.y_change = 0

    # def update_position(self, x, y):
    #     # if self.position[-1][0] != x or self.position[-1][1] != y:
    #     #     if self.food > 1:
    #     #         for i in range(0, self.food - 1):
    #     #             self.position[i][0], self.position[i][1] = self.position[i + 1]
    #     self.position = [x,y]
    #     self.x = x
    #     self.y = y

    def do_move(self, move, x, y, game, food,agent,enemy,moves,wall):
        move_array = [0, 0]

        if self.eaten:
            # self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1

        if move == 1:
            move_array = [20, 0]
        elif move == 2:  # right - going horizontal
            move_array = [0, 20]
        elif move == 3:  # right - going vertical
            move_array = [-20, 0]
        elif move == 4:  # left - going horizontal
            move_array = [0, -20]

        # if np.array_equal(move ,[1, 0, 0]):
        #     move_array = self.x_change, self.y_change
        # elif np.array_equal(move,[0, 1, 0]) and self.y_change == 0:  # right - going horizontal
        #     move_array = [0, self.x_change]
        # elif np.array_equal(move,[0, 1, 0]) and self.x_change == 0:  # right - going vertical
        #     move_array = [-self.y_change, 0]
        # elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
        #     move_array = [0, -self.x_change]
        # elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
        #     move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        # print(move_array,self.x_change,self.y_change)
        self.x = x + self.x_change
        self.y = y + self.y_change
        player_arr = [self.x,self.y]
        if self.x < 40 or self.x > game.game_width-60 or self.y < 40 or self.y > game.game_height-60 or player_arr in enemy.pos:
            game.crash = True
        enemy.do_move(moves[0],moves[1],moves[2])
        if player_arr in enemy.pos:
        	game.crash = True
        eat(self, food, game,wall)
        self.position = [[self.x,self.y]]
        # self.update_position(self.x, self.y)

    def display_player(self, x, y, game):

        # if game.crash == False:
            # for i in range(food):
            #     x_temp, y_temp = self.position[len(self.position) - 1 - i]
        game.gameDisplay.blit(self.image, (x, y))

        update_screen()
        # else:
        if game.crash:
        	pygame.time.wait(300)

#20 to 400 in both width and height
class Walls(object):
    def __init__(self):
        self.pos = []
        self.image = pygame.image.load('img/snakeBody.png')
        for i in range(7):
            self.pos.append([20+i*20,80])
        for i in range(6):
        	self.pos.append([300,40+i*20])
        for i in range(4):
        	self.pos.append([100,140+i*20])
        for i in range(3):
        	self.pos.append([220+i*20,340])
    def display_wall(self,game):
        for i in self.pos:
            game.gameDisplay.blit(self.image, (i[0], i[1]))
        update_screen()

class Enemy(object):
    def __init__(self):
        self.pos = [[40, 60], [200,40], [360,360]]
        self.image = pygame.image.load('img/enemy.png')
        self.x_change = [20, 0 ,0]
        self.y_change = [0, 20 ,-20]

    def do_move(self, move1, move2, move3):
        # move_array = [self.x_change, self.y_change]

        # if self.eaten:

        #     self.position.append([self.x, self.y])
        #     self.eaten = False
        #     self.food = self.food + 1
        if move1 == 1:
            move1_array = [20, 0]
        elif move1 == 2:  # right - going horizontal
            move1_array = [0, 20]
        elif move1 == 3:  # right - going vertical
            move1_array = [-20, 0]
        elif move1 == 4:  # left - going horizontal
            move1_array = [0, -20]
        # elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
        #     move_array = [self.y_change, 0]
        self.x_change[0], self.y_change[0] = move1_array
        if move2 == 1:
            move2_array = [20, 0]
        elif move2 == 2:  # right - going horizontal
            move2_array = [0, 20]
        elif move2 == 3:  # right - going vertical
            move2_array = [-20, 0]
        elif move2 == 4:  # left - going horizontal
            move2_array = [0, -20]
        # elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
        #     move_array = [self.y_change, 0]
        self.x_change[1], self.y_change[1] = move2_array
        if move3 == 1:
            move3_array = [20, 0]
        elif move3 == 2:  # right - going horizontal
            move3_array = [0, 20]
        elif move3 == 3:  # right - going vertical
            move3_array = [-20, 0]
        elif move3 == 4:  # left - going horizontal
            move3_array = [0, -20]
        # elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
        #     move_array = [self.y_change, 0]
        self.x_change[2], self.y_change[2] = move3_array
        for i in range(3):
            self.pos[i][0] += self.x_change[i]
            self.pos[i][1] += self.y_change[i]
            if self.pos[i][1] > 380:
            	self.pos[i][1] = 380
            if self.pos[i][1] < 40:
            	self.pos[i][1] = 40
            if self.pos[i][0] > 380:
            	self.pos[i][0] = 380
            if self.pos[i][0] < 40:
            	self.pos[i][0] = 40
            
        # self.x = x + self.x_change
        # self.y = y + self.y_change

        # if self.x < 20 or self.x > game.game_width-40 or self.y < 20 or self.y > game.game_height-40 or [self.x, self.y] in self.position:
        #     game.crash = True
        # eat(self, food, game)

        # self.update_position(self.x, self.y)

    def display_enemy(self,game):
        for i in range(3):
            game.gameDisplay.blit(self.image, (self.pos[i][0], self.pos[i][1]))
        update_screen()


class Food(object):

    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player, wall):
        x_rand = randint(40, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(40, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        while [self.x_food,self.y_food] in wall.pos:
        	x_rand = randint(40, game.game_width - 40)
	        self.x_food = x_rand - x_rand % 20
	        y_rand = randint(40, game.game_height - 40)
	        self.y_food = y_rand - y_rand % 20
        if not (self.x_food == player.x and self.y_food == player.y):
            return self.x_food, self.y_food
        else:
            self.food_coord(game,player,wall)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game,wall):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player,wall)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record, walls, enemy):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    food.display_food(food.x_food, food.y_food, game)
    walls.display_wall(game)
    enemy.display_enemy(game)
    player.display_player(player.x, player.y, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food, agent, enemy, wall):
    state_init1 = agent.get_state(game, player, food,enemy, wall)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = 1
    first_moves_enemy = [1,2,2]
    player.do_move(action, player.x, player.y, game, food, agent,enemy,first_moves_enemy,wall)
    state_init2 = agent.get_state(game, player, food,enemy,wall)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    # agent.replay_new(agent.memory,agent.more_info)
    agent.replay_new(agent.memory)


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()

def run():
    pygame.init()
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot =[]
    record = 0
    max_val = 0
    temp_score = []
    # agent.epsilon = 150
    while counter_games < 250:
        # Initialize classes
        game = Game(440, 440)
        # print(W,H)
        # agent.epsilon *= 0.98
        player1 = game.player
        food1 = game.food
        wall = game.wall
        enemy = game.enemy

        # Perform first move
        initialize_game(player1, game, food1, agent,enemy,wall)
        if display_option:
            display(player1, food1, game, record, wall, enemy)

        while not game.crash:
            #agent.epsilon is set to give randomness to actions
            agent.epsilon = 80 - counter_games
            
            #get old state
            state_old = agent.get_state(game, player1, food1,enemy,wall)
            
            #perform random actions based on agent.epsilon, or choose the action
            if randint(0, 200) < agent.epsilon:
                final_move = randint(1,4)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1,20)))
                final_move = np.argmax(prediction[0])+1
              
            # if randint(0, 200) < agent.epsilon:
            # 	possible = [1,2,3,4]
            # 	if [player1.x-20,player1.y] in wall.pos:
            # 		possible.remove(3)
            # 	if [player1.x+20,player1.y] in wall.pos:
            # 		possible.remove(1)
            # 	if [player1.x,player1.y-20] in wall.pos:
            # 		possible.remove(4)
            # 	if [player1.x,player1.y+20] in wall.pos:
            # 		possible.remove(2)
            # 	direct = randint(0,len(possible)-1)
            # 	final_move = possible[direct]
            # else:
            #     # predict action based on the old state
            #     prediction = agent.model.predict(state_old.reshape((1,20)))
            #     temp_move = (-prediction).argsort()
            #     # print(temp_move)
            #     final_move = 1
            #     for i in temp_move[0]:
            #     	if i==0:
            #     		if [player1.x+20,player1.y] in wall.pos:
            #     			continue
            #     		else:
            #     			final_move = i+1
            #     			break
            #     	if i==1:
            #     		if [player1.x,player1.y+20] in wall.pos:
            #     			continue
            #     		else:
            #     			final_move = i+1
            #     			break
            #     	if i==2:
            #     		if [player1.x-20,player1.y] in wall.pos:
            #     			continue
            #     		else:
            #     			final_move = i+1
            #     			break
            #     	if i==3:
            #     		if [player1.x,player1.y-20] in wall.pos:
            #     			continue
            #     		else:
            #     			final_move = i+1
            #     			break
            #     # final_move = np.argmax(prediction[0])+1
            #     # print(state_old,final_move)
            # # print(state_old,final_move)
            #     # print(final_move)
            # # print(final_move)
            #perform new move and get new state
            
            moves = []
            for i in range(3):
                possible = [1,2,3,4]
                if enemy.pos[i][0]-20 < 40 or [enemy.pos[i][0]-20,enemy.pos[i][1]] in wall.pos:
                    possible.remove(3)
                if enemy.pos[i][0]+20 > game.game_width-60 or [enemy.pos[i][0]+20,enemy.pos[i][1]] in wall.pos:
                    possible.remove(1)
                if enemy.pos[i][1]-20 < 40 or [enemy.pos[i][0],enemy.pos[i][1]-20] in wall.pos:
                    possible.remove(4)
                if enemy.pos[i][1]+20 > game.game_height-60 or [enemy.pos[i][0],enemy.pos[i][1]+20] in wall.pos:
                    possible.remove(2)
                # print(possible)
                direct = randint(0,len(possible)-1)
                moves.append(possible[direct])
                    # game.crash = True
        
            # enemy.do_move(moves[0],moves[1],moves[2])

            player1.do_move(final_move, player1.x, player1.y, game, food1, agent,enemy,moves,wall)
            state_new = agent.get_state(game, player1, food1, enemy, wall)

            # print(state_old,final_move,state_new)
            
            #set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            
            #train short memory base on the new action and state
            agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
            
            # store the new data into a long term memory
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            record = get_record(game.score, record)
            if display_option:
                display(player1, food1, game, record, wall, enemy)
                pygame.time.wait(speed)
        # print(enemy.pos)
        # print(agent.epsilon)
        agent.replay_new(agent.memory)
        # agent.replay_new(agent.memory,agent.more_info)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        # print('More Info size: ', len(agent.more_info), 'memory size: ', len(agent.memory))
        if len(temp_score) == 5:
            temp_score = temp_score[1:]
        temp_score.append(game.score)
        score_val = sum(temp_score)/len(temp_score)
        score_plot.append(score_val)
        counter_plot.append(counter_games)
        print(score_plot)
        print(prediction,final_move)
        if score_val > max_val:
            max_val = score_val
            agent.model.save_weights('pacman_base.hdf5')
        # print(player1.position)
    # agent.model.save_weights('weights_pacman.hdf5')
    # plot_seaborn(counter_plot, score_plot)
    plt.plot(counter_plot, score_plot)
    # plt.show()
    plt.savefig('pacman_base.png')


run()
