import numpy as np
import matplotlib.pyplot as plt

class Dealer:
	def __init__(self):
		"""
		The policy depends on the sum of cards with the dealer.
		index:0 -- sum:12
		index:10 -- sum:21
		If the sum>17, the dealer sticks else hits
		"""
		self.policy = np.zeros([10])
		self.actions = {'hit':0, 'stick':1}
		self.policy[5:] = self.actions['stick']

	def take_action(self, state):
		"""
		state = {0,1,...,10} -> {12,13,....,21}
		"""		
		if state < 0:
			return self.actions['hit']

		return self.policy[state]


class MC_Agent:
	"""
	There are 200 states possible.
	There are three variables with the specified valid ranges:
		1) S: The sum of cards in the agents hand (12-21)
		2) D: The value of the showing card by the dealer (Ace-10)
		3) U: Whether the agent has an usable ace (0-1)

	State Values v(s): specified by a three-dimensional array of shape SxDxU.
	
	Action Values Q(s,a): specified by the four dimensional array 
	action_values whose first three dimensions specify the state and 
	the fourth dimension specifies the action taken.
	
	For S<12, the agent should always hit.
	For S>21, the agent will go burst.
	
	Reward Scheme:
		1) Win:  +1
		2) Draw:  0
		3) Lose: -1
		4) Rest of the states in a game: 0	
	"""
	def __init__(self):
		self.policy = np.zeros([10,10,2])
		self.actions = {'hit':0, 'stick':1}
		self.Q = np.zeros([10,10,2,2])		
		self.V = np.zeros([10,10,2])
		self.state_frequency = np.zeros([10,10,2])
		self.state_action_frequency = np.zeros([10,10,2,2])		
		self.gamma = 1

	def load_data(self, path):
		self.Q = np.load(path+'Q.npy')
		self.policy = np.load(path+'policy.npy')
		self.V = np.load(path+'V.npy')

	def set_policy(self, new_policy):
		"""
		new_policy: A (10,10,2) matrix
		"""
		self.policy = new_policy

	def first_visit_MC_prediction(self):
		"""
		Used to estimate the state values for a given policy.
		"""
		for i in range(500000):
			env = Environment(self)
			states, actions, rewards = env.play()			
			G = 0								

			if states!=None:
				state_visited = self.create_state_visited_array(states)			
				for s,r in zip(reversed(states), reversed(rewards)):
					G = self.gamma * G + r
					if state_visited[s]==1:												
						self.V[s] = (self.V[s] * self.state_frequency[s] +  G) \
									/ (self.state_frequency[s] + 1)
						self.state_frequency[s] += 1															
					state_visited[s] -= 1

			# print(states)
			# print(actions)
			# print(rewards)
			# print(self.V)
			
			if i%1000==0:
				print('i=',i)
				self.plot(data=self.V[:,:,0], \
						  datatype='values', \
						  figname='./first_visit_MC_prediction/no_usable_ace/' + str(i), \
						  title='V i='+str(i))
				self.plot(data=self.V[:,:,1], \
				 		  datatype='values', \
				 		  figname='./first_visit_MC_prediction/usable_ace/' + str(i), \
				 		  title='V i='+str(i))
				np.save('values',self.V)

	def MC_exploring_starts(self):
		"""
		For MC Control to find the optimal policy
		and the optimal state-action values.
		"""		
		for i in range(500000,1000000):
			initial_state = self.generate_random_state()
			initial_action = self.generate_random_action()
			env = Environment(self)
			states, actions, reward = env.play(initial_state, initial_action)			
			G = 0

			# print('states = ', states)
			# print('actions = ', actions)
			# print('rewards = ', reward)						
						
			state_action_visited = self.create_state_action_visited_array(states, actions)
			rewards = [0] * len(states)
			rewards[-1] = reward			
			for s,a,r in zip(reversed(states), reversed(actions), reversed(rewards)):				
				sa = s+(a,)
				G = self.gamma * G + r				
				if state_action_visited[sa] == 1:						
					self.Q[sa] = (self.Q[sa] * self.state_action_frequency[sa] +  G) \
								 / (self.state_action_frequency[sa] + 1)
					self.state_action_frequency[sa] += 1														   
					self.policy[s] = np.argmax(self.Q[s])
				state_action_visited[sa] -= 1

			self.find_V_from_Q()

			if i%1000==0:
				print('i=',i)
				self.plot(data=self.V[:,:,0], datatype='values', figname='./MC_exploring_starts/no_usable_ace/V/' + str(i), title='V i='+str(i))
				self.plot(data=self.V[:,:,1], datatype='values', figname='./MC_exploring_starts/usable_ace/V/' + str(i), title='V i='+str(i))
				self.plot(data=self.policy[:,:,0], datatype='policy', figname='./MC_exploring_starts/no_usable_ace/policy/' + str(i), title='policy i='+str(i))
				self.plot(data=self.policy[:,:,1], datatype='policy', figname='./MC_exploring_starts/usable_ace/policy/' + str(i), title='policy i='+str(i))

				np.save('./MC_exploring_starts/V', self.V)
				np.save('./MC_exploring_starts/Q', self.Q)											    
				np.save('./MC_exploring_starts/policy', self.policy)

	def find_V_from_Q(self):
		"""
		The action selected by the policy has probability one,
		while the rest has a probability of zero. 
		"""
		for i in range(self.Q.shape[0]):
			for j in range(self.Q.shape[1]):
				for k in range(self.Q.shape[2]):
					l = self.policy[i,j,k]
					self.V[i,j,k] = self.Q[i,j,k,l]

	def generate_random_state(self):		
		return (np.random.randint(0,10), np.random.randint(0,10), np.random.randint(0,2))

	def generate_random_action(self):
		return np.random.randint(0,2)

	def create_state_visited_array(self, states):
		state_visited = np.zeros([10,10,2])

		for s in states:
			state_visited[s] += 1

		return state_visited

	def create_state_action_visited_array(self, states, actions):
		state_action_visited = np.zeros([10,10,2,2])

		for s,a in zip(states, actions):
			state_action = s + (a,)
			state_action_visited[state_action] += 1

		return state_action_visited

	def take_action(self, state):
		"""
		state = (player_sum, showing_card, usable_ace)				
		"""	
		if state[0] < 0:
			return self.actions['hit']

		return self.policy[state]

	def plot(self, data, datatype, figname, title):
		fig, ax = plt.subplots()
		fig.set_size_inches(10, 10)

		im = None
		if datatype=="policy":
			im = ax.imshow(data)
		elif datatype=="values":
			im = ax.imshow(data)

		cbar = ax.figure.colorbar(im, ax=ax)

		if datatype=="policy":
			cbar.ax.set_ylabel("Actions", rotation=-90, va="bottom")
		elif datatype=="values":
			cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")
		
		ax.set_xticks(np.arange(0,10))
		ax.set_yticks(np.arange(0,10))
		ax.set_xticklabels(np.arange(1,11), fontsize=14)
		ax.set_yticklabels(np.arange(12,22), fontsize=14)
		ax.set_xlabel("Showing Card", fontsize=16)
		ax.set_ylabel("Sum of Cards", fontsize=16)		

		for edge, spine in ax.spines.items():
			spine.set_visible(False)

		ax.set_xticks(np.arange(0,11)-.5, minor=True)
		ax.set_yticks(np.arange(0,11)-.5, minor=True)
		ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
		ax.tick_params(which="minor", bottom=False, left=False)				
		ax.set_title(title, fontsize=22)		

		fig.tight_layout()
		plt.savefig(figname)
		plt.close()

class Environment:
	def __init__(self, player):
		self.dealer = Dealer()
		self.player = player
		self.dealer_sum = 0
		self.player_sum = 0
		self.dealer_usable_ace = 0
		self.player_usable_ace = 0
		self.turns = 0
		self.showing_card = 0

		self.list = [1,10,10,3,10]
		self.i = 0									

	def generate_card(self):
		card = np.random.choice([1,2,3,4,5,6,7,8,9,10,10,10,10])					
		return card
		
		self.i += 1
		return self.list[self.i-1]	

	def generate_reward(self):		
		if self.player_sum==self.dealer_sum:
			return 0
		elif self.player_sum==21:
			return 1
		elif self.dealer_sum==21:
			return -1
		elif self.player_sum>21 and self.dealer_sum>21:
			return 0
		elif self.player_sum>21 and self.dealer_sum<21:
			return -1
		elif self.player_sum<21 and self.dealer_sum>21:
			return 1			
		elif self.player_sum>self.dealer_sum:
			return 1
		elif self.player_sum<self.dealer_sum:
			return -1		
		return 0

	def dealer_play(self, action=None):
		card = None		
		state = self.dealer_sum - 12			

		if action==None:
			action = self.dealer.take_action(state)			
		if action == self.dealer.actions['hit']:
			card = self.generate_card()		
			self.dealer_sum += card
			if card==1 and self.dealer_sum+10<=21:
				self.dealer_sum += 10
				self.dealer_usable_ace = 1
			if self.dealer_sum>21 and self.dealer_usable_ace==1:
				self.dealer_sum -= 10
				self.dealer_usable_ace = 0

		return card, action

	def player_play(self, action=None):
		card = None		
		state = (self.player_sum-12, self.showing_card-1, self.player_usable_ace)		
		
		if action==None:
			action = self.player.take_action(state)
		if action == self.player.actions['hit']:
			card = self.generate_card()		
			self.player_sum += card
			if card==1 and self.player_sum+10<=21:
				self.player_sum += 10
				self.player_usable_ace = 1
			if self.player_sum>21 and self.player_usable_ace==1:
				self.player_sum -= 10
				self.player_usable_ace = 0
		
		return card, action
	
	def play(self, initial_state=None, initial_action=None):
		self.reset()		
		card = None
		action = None		
		states = list()
		actions = list()

		if initial_state==None:
			initial_state = self.player.generate_random_state()
			
		action = initial_action
		self.player_sum = initial_state[0] + 12
		self.showing_card = initial_state[1] + 1
		self.dealer_sum += self.showing_card
		if self.showing_card==1:
			self.dealer_sum += 10
			self.dealer_usable_ace = 1
		self.player_usable_ace = initial_state[2]				
		
		# print('state = ',(self.player_sum, self.showing_card, self.player_usable_ace),'action = ',action)

		states.append(initial_state)

		if initial_action!=None:						
			self.player_play(initial_action)
			actions.append(initial_action)
		# print('P:', 'state = ', (self.player_sum, self.showing_card, self.player_usable_ace), 'action = ', action)		
		while self.player_sum<=21:			
			states.append((self.player_sum-12, self.showing_card-1, self.player_usable_ace))
			card, action = self.player_play()
			actions.append(action)			
			# print('P:', 'card = ', card, 'state = ', (self.player_sum, self.showing_card, self.player_usable_ace), 'action = ', action)
			if action==self.player.actions['stick']:
				break

		# print('D: state = ', (self.dealer_sum, self.dealer_usable_ace))
		while self.dealer_sum<=21:
			card, action = self.dealer_play()
			state = (self.dealer_sum, self.dealer_usable_ace)
			# print('D: card = ', card, 'state = ', state, 'action = ', action)
			if action==self.dealer.actions['stick']:
				break

		reward = self.generate_reward()

		# print('reward = ', reward)
		return states, actions, reward
						
	def print_game_status(self, player_sum, dealer_sum, player_card, dealer_card):
		print('Player Card: ', player_card)
		print('Dealer Card: ', dealer_card)
		print('Player Sum: ', player_sum)
		print('Dealer Sum: ', dealer_sum)
		print('-'*50)

	def reset(self):
		self.turns = 0
		self.dealer_sum = 0
		self.player_sum = 0
		self.player_usable_ace = 0
		self.dealer_usable_ace = 0
		self.showing_card = 0
		

if __name__ == '__main__':
	agent = MC_Agent()
	# new_policy = np.zeros((10,10,2), dtype=int)
	# new_policy[8:][:][:] = agent.actions['stick']	
	# agent.set_policy(new_policy)
	agent.load_data('.//MC_exploring_starts/')
	# agent.first_visit_MC_prediction()
	agent.MC_exploring_starts()