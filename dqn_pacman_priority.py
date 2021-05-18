from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        self.epsilon = 0
        self.actual = []
        self.memory = []
        self.more_info = []

    def get_state(self, game, player, food, enemy, wall):
        
        one_next = []
        for j in range(4):
            ene = []
            if j==0:
                for i in range(3):
                    ene.append(abs(player.x+20-enemy.pos[i][0])+abs(player.y-enemy.pos[i][1]))
            if j==1:
                for i in range(3):
                    ene.append(abs(player.x-enemy.pos[i][0])+abs(player.y+20-enemy.pos[i][1]))
            if j==2:
                for i in range(3):
                    ene.append(abs(player.x-20-enemy.pos[i][0])+abs(player.y-enemy.pos[i][1]))
            if j==3:
                for i in range(3):
                    ene.append(abs(player.x-enemy.pos[i][0])+abs(player.y-20-enemy.pos[i][1]))
            if min(ene)/20 == 1:
                one_next.append(1)
            else:
                one_next.append(0)


        # a = list(map(add, player.position[-1], [20, 0]))
        state = [
            ((list(map(add, player.position[-1], [20, 0])) in enemy.pos) or player.x + 20 > game.game_width-40),  # danger right
            ((list(map(add, player.position[-1], [0, 20])) in enemy.pos) or player.y + 20 > game.game_height-40),  # danger down
            ((list(map(add, player.position[-1], [-20, 0])) in enemy.pos) or player.x - 20 < 20), #danger left
            ((list(map(add, player.position[-1], [0, -20])) in enemy.pos) or player.y - 20 < 20), #danger up

            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y,  # food down
            one_next[0],
            one_next[1],
            one_next[2],
            one_next[3],
            (list(map(add, player.position[-1], [20, 0])) in wall.pos),
            (list(map(add, player.position[-1], [0, 20])) in wall.pos),
            (list(map(add, player.position[-1], [-20, 0])) in wall.pos),
            (list(map(add, player.position[-1], [0, -20])) in wall.pos)
            # enemy.pos[ind][0] < player.x, # nearest enemy left
            # enemy.pos[ind][0] > player.x, # nearest enemy right
            # enemy.pos[ind][1] < player.y, # nearest enemy up
            # enemy.pos[ind][1] > player.y # nearest enemy down
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -1
            return self.reward
        if player.eaten:
            self.reward = 1
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=20))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=4, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        # print(reward)
        if reward == 0:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.more_info.append((state, action, reward, next_state, done))

    def replay_new(self, memory, more_info):
        big_mem = memory+more_info
        if len(big_mem) > 1000:
            temp = len(more_info)
            # minibatch = []
            minibatch = random.sample(more_info,min(int(temp*0.7),700))
            minibatch += random.sample(memory, 1000-min(int(temp*0.7),700))
        else:
            minibatch = big_mem
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action-1] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 20)))[0])
        if target > 0.9 or target < -0.2:
            self.more_info.append((state, action, target, next_state, done))
        else:
            self.memory.append((state,action,target,next_state,done))
        target_f = self.model.predict(state.reshape((1, 20)))
        target_f[0][action-1] = target
        self.model.fit(state.reshape((1, 20)), target_f, epochs=1, verbose=0)