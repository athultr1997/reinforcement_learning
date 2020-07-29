import numpy as np
import matplotlib.pyplot as plt
import os

Actions = {'LEFT':0,'UP':1,'RIGHT':2,'DOWN':3, 'LEFTUP':4, 'LEFTDOWN':5, 'RIGHTUP':6, 'RIGHTDOWN':7}

class SimpleEnvironment:
	def __init__(self):		
		self.gridworld = np.zeros((7,10))		
		self.terminal_state = (3,7)
		self.initial_state = (3,0)		
		self.current_state = self.initial_state		
		# self.timesteps = 0
		# self.episode = 0

		# self.path = './games/' + str(self.episode) + '/'		
		# if not os.path.exists(self.path):
		# 	os.makedirs(self.path)

		# self.gridworld[self.initial_state] = 2
		# self.gridworld[self.terminal_state] = 3
		
		# self.plot(self.initial_state)

	def calculate_next_state(self, action):
		state = list(self.current_state)
		
		if action == Actions['LEFT']:
			state[1] -= 1
		elif action == Actions['UP']:
			state[0] -= 1
		elif action == Actions['RIGHT']:
			state[1] += 1
		elif action == Actions['DOWN']:
			state[0] += 1

		if (state[0] >= 0 and state[0] < self.gridworld.shape[0]) and (state[1] >= 0 and state[1] < self.gridworld.shape[1]):
			state = tuple(state)
			self.current_state = state
		else:
			state = self.current_state

		# self.timesteps += 1
		# self.plot(state)		

		return state

	def calculate_reward(self, state):
		if state == self.terminal_state:
			return 0
		else:
			return -1		

	def play_step(self, action):
		new_state = self.calculate_next_state(action)
		reward = self.calculate_reward(new_state)		
		return new_state, reward

	def return_initial_state(self):
		return self.initial_state

	def return_terminal_state(self):
		return self.terminal_state

	def plot(self, state):
		self.gridworld[state] = 1
		fig, ax = plt.subplots()
		fig.set_size_inches(10, 10)

		im = None		
		im = ax.imshow(self.gridworld)		
				
		ax.set_xticks(np.arange(0, self.gridworld.shape[1]))
		ax.set_yticks(np.arange(0, self.gridworld.shape[0]))
		ax.set_xticklabels(self.winds, fontsize=14)
		ax.set_yticklabels([])				

		for edge, spine in ax.spines.items():
			spine.set_visible(False)

		ax.set_xticks(np.arange(0, self.gridworld.shape[1]+1)-.5, minor=True)
		ax.set_yticks(np.arange(0, self.gridworld.shape[0]+1)-.5, minor=True)
		ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
		ax.tick_params(which="minor", bottom=False, left=False)
		ax.set_title('Game: ' + str(self.episode) + '; Timestep: ' + str(self.timesteps), fontsize=22)

		fig.tight_layout()
		plt.savefig(self.path + str(self.timesteps))
		plt.close()

		if state == self.initial_state:
			self.gridworld[state] = 2
		elif state == self.terminal_state:
			self.gridworld[state] = 3
		else:
			self.gridworld[state] = 0

	def reset(self):
		self.current_state = self.initial_state		
		# self.episode += 1
		# self.timesteps = 0

class WindyEnvironment:
	def __init__(self, grid_shape, num_actions):
		self.grid_shape = grid_shape
		self.num_actions = num_actions
		self.gridworld = np.zeros(grid_shape)
		self.terminal_state = (3,7)
		self.initial_state = (3,0)
		self.current_state = self.initial_state
		self.winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
		self.timesteps = 0
		self.episode = 0

		# self.path = './games/' + str(self.episode) + '/'		
		# if not os.path.exists(self.path):
		# 	os.makedirs(self.path)

		# self.gridworld[self.initial_state] = 2
		# self.gridworld[self.terminal_state] = 3
		
		# self.plot(self.initial_state)

	def calculate_next_state(self, action):		
		state = list(self.current_state)

		wind = self.winds[state[1]]
		
		if action == Actions['LEFT']:
			state[1] -= 1
		elif action == Actions['UP']:
			state[0] -= 1
		elif action == Actions['RIGHT']:
			state[1] += 1
		elif action == Actions['DOWN']:
			state[0] += 1

		state[0] = min(self.grid_shape[0]-1, state[0])
		state[0] = max(0, state[0])

		state[1] = min(self.grid_shape[1]-1, state[1])
		state[1] = max(0, state[1])
		
		state[0] = max(0, state[0] - wind)	

		state = tuple(state)

		self.current_state = state
		
		self.timesteps += 1
		# self.plot(state)		

		return state

	def calculate_reward(self, state):
		if state == self.terminal_state:
			return 0
		else:
			return -1

	def play_step(self, action):
		new_state = self.calculate_next_state(action)
		reward = self.calculate_reward(new_state)		
		return new_state, reward

	def return_initial_state(self):
		return self.initial_state

	def return_terminal_state(self):
		return self.terminal_state

	def return_timesteps(self):
		return self.timesteps

	def return_grid_specs(self):
		return self.grid_shape, self.num_actions

	def plot(self, state):
		self.gridworld[state] = 1
		fig, ax = plt.subplots()
		fig.set_size_inches(10, 10)

		im = None		
		im = ax.imshow(self.gridworld)		
				
		ax.set_xticks(np.arange(0, self.grid_shape[1]))
		ax.set_yticks(np.arange(0, self.grid_shape[0]))
		ax.set_xticklabels(self.winds, fontsize=14)
		ax.set_yticklabels([])				

		for edge, spine in ax.spines.items():
			spine.set_visible(False)

		ax.set_xticks(np.arange(0, self.grid_shape[1]+1)-.5, minor=True)
		ax.set_yticks(np.arange(0, self.grid_shape[0]+1)-.5, minor=True)
		ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
		ax.tick_params(which="minor", bottom=False, left=False)
		ax.set_title('Game: ' + str(self.episode) + '; Timestep: ' + str(self.timesteps), fontsize=22)

		fig.tight_layout()
		plt.savefig(self.path + str(self.timesteps))
		plt.close()

		if state == self.initial_state:
			self.gridworld[state] = 2
		elif state == self.terminal_state:
			self.gridworld[state] = 3
		else:
			self.gridworld[state] = 0

	def reset(self):
		self.current_state = self.initial_state		
		self.episode += 1
		self.timesteps = 0

if __name__ == '__main__':
	env = WindyEnvironment()
	new_state, reward = env.play_step(3)
	print()
	print(new_state)