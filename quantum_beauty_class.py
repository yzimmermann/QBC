import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

np.random.seed(42)
random.seed(42)

class Player:
    def __init__(self, num_states, mutation_rate, mutation_strength) -> None:
        
        self.state = np.random.rand(num_states)
        self.state = self.state/np.linalg.norm(self.state)
        self.num_states = num_states
        self.mutation_rate =  mutation_rate
        self.mutation_strength = mutation_strength
        self.age = 0
        self.weights = self.comp_weights()


        pass

    def update_state(self, parent_1, parent_2):
        #make a child
        self.state = parent_1.state + parent_2.state
        self.state /= np.linalg.norm(self.state)

        #Add some mutation
        for i in range(self.num_states):
            if np.random.random() < self.mutation_rate:
                self.state[i] += np.random.uniform(-self.mutation_strength, self.mutation_strength)
                self.state[i] = max(0, self.state[i])
        self.state /= np.linalg.norm(self.state)
        self.weights = self.comp_weights
        self.age = 0

    def comp_weights(self):
        return np.multiply(self.state, np.conjugate(self.state))

    def comp_exp_val(self):
        return np.dot(np.arange(self.num_states), self.weights)
    
    def comp_std_val(self):
        return np.sqrt(np.average((np.arange(self.num_states)-self.comp_exp_val())**2, weights=self.weights))


class qm_beautycontest:

    def __init__(self, num_players, num_states, mutation_rate, mutation_strength, contr_factor, fitness_type, num_elite) -> None:

        self.players = [Player(num_states, mutation_rate, mutation_strength) for _ in range(num_players)]
        self.num_players = num_players
        self.num_states = num_states
        self.p = contr_factor
        assert isinstance(fitness_type, str)
        self.fitness_type = fitness_type
        self.num_elite  = num_elite

    
    def fitness(self):
        self.fit = np.zeros(self.num_players)

        if self.fitness_type == 'single':
            for i in range(self.num_players):
                self.fit[i] = 1/(np.linalg.norm(self.players[i].state[self.measured_state])**2 + 1)
        
        elif self.fitness_type == 'avg':
            for i in range(self.num_players):
                self.fit[i] = np.abs(self.measured_state - self.players[i].comp_exp_val())

        elif self.fitness_type == 'measured':
            #collapse the state of the player and give the fit by the distance to the measured state
            for i in range(self.num_players):
                prob_player = self.players[i].weights
                self.fit[i] = np.abs(self.measured_state - np.random.choice(range(self.num_states), p=prob_player))

        else:
            for i in range(self.num_players):
                self.fit[i] = 1/(np.linalg.norm(self.players[i].state[self.measured_state])**2 + 1)


    def play_round(self):
        #Get the total state, draw the winning number, multiply by p and round
        self.tot_state = np.zeros(self.num_states)
        for i in range(self.num_players):
            self.tot_state += self.players[i].state

        self.tot_state /= np.linalg.norm(self.tot_state)
        self.prob_vec = np.multiply(self.tot_state, np.conjugate(self.tot_state))

        self.measured_state = int(np.round(np.random.choice(range(self.num_states), p=self.prob_vec) * self.p))
        if self.measured_state >= self.num_states:
            self.measured_state = self.num_states - 1
        elif self.measured_state < 0:
            self.measured_state = 0

        self.fitness()

        #determine the elite by the lowest fitness scores
        def get_n_smallest_indices(arr, n):
            indices = np.argpartition(arr, n)[:n]
            return indices
        
        def get_remaining_indices(arr, n):
            indices = np.argpartition(arr, n)[n:]
            return indices

        sort = np.argsort(self.fit)
        self.elite = sort[:self.num_elite]
        self.loosers = sort[self.num_elite:]
        self.winner = sort[0]

        self.elite_states = [self.players[i] for i in self.elite]

        for i in self.elite:
            self.players[i].age += 1

        for i in self.loosers:
            p_1, p_2 = random.choices(self.elite_states, k=2)
            self.players[i].update_state(p_1, p_2)
