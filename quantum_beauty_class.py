import numpy as np
import scipy as sp


class Player:
    def __init__(self, num_states, mutation_rate, mutation_strength) -> None:
        
        self.state = np.random.rand(num_states)
        self.state = self.state/np.linalg.norm(self.state)
        self.num_states = num_states
        self.mutation_rate =  mutation_rate
        self.mutation_strength = mutation_strength

        pass

    def update_state(self, parent_1, parent_2):
        #make a child
        self.state = parent_1 + parent_2
        self.state /= np.linalg.norm(self.state)

        #Add some mutation
        for i in range(self.num_states):
            if np.random.random() < self.mutation_rate:
                self.state[i] += np.random.uniform(-self.mutation_strength, self.mutation_strength)
                self.state[i] = max(0, self.state[i])
        self.state /= np.linalg.norm(self.state)


class qm_beautycontest:

    def __init__(self, num_players, num_states, mutation_rate, mutation_strength, contr_factor) -> None:

        self.players = [Player(num_states, mutation_rate, mutation_strength) for _ in range(num_players)]
        self.num_players = num_players
        self.num_states = num_states
        self.p = contr_factor


    def play_round(self):
        #Get the total state, draw the winning number, multiply by p and round
        tot_state = np.zeros(self.num_states)
        for i in range(self.num_players):
            tot_state += self.players[i].state

        tot_state /= np.linalg.norm(tot_state)
        prob_vec = np.linalg.norm(tot_state, axis=-1)**2



