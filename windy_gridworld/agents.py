import numpy as np
import matplotlib.pyplot as plt
import os
from environments import SimpleEnvironment

Actions = {'LEFT':0,'UP':1,'RIGHT':2,'DOWN':3, 'LEFTUP':4, 'LEFTDOWN':5, 'RIGHTUP':6, 'RIGHTDOWN':7}

class TDAgent:
	def __init__(self):
		self.alpha = 0.1
		self.epsilon = 0.1
		self.gamma = 1

		if not os.path.exists('./policy/'):
			os.makedirs('./policy/')

		if not os.path.exists('./V/'):
			os.makedirs('./V/')

	def TD0_update(self, state, action, reward, new_state, new_action):		
		state_action = state + (action,)
		new_state_action = new_state + (action,)		
		self.Q[state_action] += self.alpha * (reward + (self.gamma * self.Q[new_state_action]) - self.Q[state_action])

	def greedy_action(self, state):
		return np.argmax(self.Q[state])

	def random_action(self, state):
		return Actions[np.random.choice(['LEFT', 'UP', 'RIGHT', 'DOWN'])]

	def choose_action(self, state):
		action = None
		rand = np.random.uniform(low = 0.0, high = 1.0, size = None)
		if rand >= self.epsilon:
			action = self.greedy_action(state)
		else:
			action = self.random_action(state)
		return action

	def update_V_from_Q(self):
		self.V = np.amax(self.Q, axis=2)

	def update_policy_from_Q(self):
		for i in range(self.policy.shape[0]):
			for j in range(self.policy.shape[1]):
				self.policy[i,j] = np.argmax(self.Q[i,j])

	def SARSA(self, env):
		grid_shape, num_actions = env.return_grid_specs()

		self.Q = np.zeros(grid_shape + (num_actions,))
		self.V = np.zeros(grid_shape)
		self.policy = np.zeros(grid_shape, dtype=np.int)
		
		for i in range(200):
			env.reset()			

			self.initial_state = env.return_initial_state()
			self.terminal_state = env.return_terminal_state()
			state = self.initial_state
			action = self.choose_action(state)
			new_state = None
			new_action = None

			print('i = ', i)			

			while state != self.terminal_state:						
				new_state, reward = env.play_step(action)
				new_action = self.choose_action(new_state)
				self.TD0_update(state, action, reward, new_state, new_action)
				state = new_state
				action = new_action
				# print('state = ', state, '; action = ', action, '; reward = ', reward)

			self.update_V_from_Q()
			self.update_policy_from_Q()

			self.plot(datatype='policy', title='Policy (iter='+str(i)+')', figname='./policy/'+str(i))
			self.plot(datatype='values', title='V (iter='+str(i)+')', figname='./V/'+str(i))
			np.save('V', self.V)
			np.save('Q', self.Q)
			np.save('policy', self.policy)

			print('timesteps = ', env.return_timesteps())
			print('-' * 50)

	def plot(self, datatype, title, figname):
		fig, ax = plt.subplots()
		fig.set_size_inches(10, 10)

		im = None
		if datatype == "policy":
			grid = np.full(self.policy.shape, -1)
			grid[self.initial_state] = 1
			grid[self.terminal_state] = 2
			im = ax.imshow(grid, cmap='PiYG')	
		elif datatype == "values":
			im = ax.imshow(self.V)		

		if datatype=="values":
			cbar = ax.figure.colorbar(im, ax=ax)
			cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")
		
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		for edge, spine in ax.spines.items():
			spine.set_visible(False)

		Arrows = {0:r'$\leftarrow$', 1:r'$\uparrow$', 2:r'$\rightarrow$', 3:r'$\downarrow$', \
				  4:r'$\nwarrow$', 5:r'$\swarrow$', 6:r'$\nearrow$', 7:r'$\searrow$'}

		if datatype=="policy":
			for i in range(self.policy.shape[0]):
				for j in range(self.policy.shape[1]):
					ax.text(j, i, Arrows[self.policy[i][j]], ha="center", va="center", color="black", fontsize=30)

		ax.set_xticks(np.arange(0,self.policy.shape[1]+1)-.5, minor=True)
		ax.set_yticks(np.arange(0,self.policy.shape[0]+1)-.5, minor=True)
		if datatype == "policy":
			ax.grid(which="minor", color="black", linestyle='-', linewidth=3)
		elif datatype == "values":
			ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
		ax.tick_params(which="minor", bottom=False, left=False)				
		ax.set_title(title, fontsize=22)		

		fig.tight_layout()
		plt.savefig(figname)
		plt.close()