import numpy as np
import matplotlib.pyplot as plt

"""
Definitions
-----------
Actions are indexed from '0' to 'N-1' if there are 'N' actions
"""

class epsilon_greedy_agent:
	def __init__(self, N = 10, epsilon = 0.1, initial_value = 0):
		"""
		table is the DS which stores the values for each actions
		table[i][0] -> The estimate of the value of action 'i'
		table[i][1] -> Sum of rewards when action 'i' was taken prior to time 't'
		table[i][2] -> No:of times action 'i' was taken prior to time 't'
		"""
		self.N = N
		self.epsilon = epsilon
		self.initial_value = initial_value
		self.table = np.full([self.N,3], self.initial_value, dtype = float)

	def act(self):
		s = np.random.uniform(low = 0.0, high = 1.0, size = None)

		if s > self.epsilon:
			maxima = np.amax(self.table[:,0])
			max_indices = np.where(self.table[:,0] == maxima)[0]
			if max_indices.size == 1:
				return int(max_indices[0])
			else:
				# Tie-breaking
				p = np.random.uniform(low = 0.0, high = max_indices.size, size = None)
				return int(max_indices[int(p)])
		else:
			l = np.random.uniform(low = 0.0, high = self.N, size = None)
			return int(l)

	def update_table(self,action,reward):
		self.table[action][1] += reward
		self.table[action][2] += 1
		self.table[action][0] = self.table[action][1] / self.table[action][2]

	def reset(self):
		self.table = np.full([self.N,3], self.initial_value, dtype = float) 		
		

class bandit_problem_test_bed:
	def __init__(self, N):
		self.N = N
		self.true_values = None
		self.optimal_action = None

	def reset(self):
		self.true_values = np.random.normal(loc = 0.0, scale = 1.0, size = self.N)
		self.optimal_action = np.where(self.true_values == np.amax(self.true_values))[0][0]
		# plt.scatter(range(self.N), self.true_values, marker = 'o', c = 'r')
		# plt.show()

	def calculate_reward(self, action):
		return np.random.normal(loc = self.true_values[action], scale = 1.0, size = None)

	# def plot_init(self):
		# plt.figure(num=1,figsize=(12, 9))
		# for spine in plt.gca().spines.values():
  #   		spine.set_visible(False)


		# plt.figure(num=2,figsize=(12, 9))
		# for spine in plt.gca().spines.values():
  #   		spine.set_visible(False)



	def plot(self, avg_rewards_list, opt_action_per_list, agent):
		plt.figure(1)
		plt.plot(range(self.time_steps),avg_rewards_list,label = "$\epsilon$ = " + str(agent.epsilon) + "; initial value = " + str(agent.initial_value))
		plt.figure(2)
		plt.plot(range(self.time_steps),opt_action_per_list,label = "$\epsilon$ = " + str(agent.epsilon) + "; initial value = " + str(agent.initial_value))

	# def show_figures(self):
	# 	plt.figure(1)
	# 	plt.title("Average Rewards VS Time Step Graph for " + str(self.N) + " Bandits")
	# 	plt.ylabel("Average Rewards for " + str(self.runs) + " Runs")
	# 	plt.xlabel("Time Step")
	# 	plt.legend()		

	# 	plt.figure(2)
	# 	plt.title("Percentage Optimal Action VS Time Step Graph for " + str(self.N) + " Bandits")
	# 	plt.ylabel("Percentage Optimal Action for " + str(self.runs) + " Runs")
	# 	plt.xlabel("Time Step")
	# 	plt.legend()

	# 	plt.show()

	def show_figures(self):
		plt.figure(num=1,figsize=(12, 9))
		plt.gca().spines["right"].set_visible(False)
		plt.gca().spines["top"].set_visible(False)
		plt.gca().spines["bottom"].set_visible(True)
		plt.gca().spines["left"].set_visible(True)				
		plt.gca().get_xaxis().tick_bottom()
		plt.gca().get_yaxis().tick_left()
		plt.ylim(0,1.8)
		plt.xlim(0,1000)
		plt.xticks(np.arange(0,1001,100),fontsize=14)
		plt.yticks(np.arange(0,1.81,0.2),fontsize=14)
		plt.title("Average Reward Plot for " + str(self.N) + " Bandits",fontsize=22)
		plt.ylabel("Average Rewards for " + str(self.runs) + " Runs",fontsize=16)
		plt.xlabel("Time Steps",fontsize=16)
		plt.legend(loc="upper left",fontsize=14)
		plt.savefig("Average Rewards",bbox_inches="tight")		

		plt.figure(num=2,figsize=(12, 9))
		plt.gca().spines["right"].set_visible(False)
		plt.gca().spines["top"].set_visible(False)
		plt.gca().spines["left"].set_visible(True)
		plt.gca().spines["bottom"].set_visible(True)				
		plt.gca().get_xaxis().tick_bottom()
		plt.gca().get_yaxis().tick_left()
		plt.ylim(0,100)
		plt.xlim(0,1000)
		plt.xticks(np.arange(0,1001,100),fontsize=14)
		plt.yticks(np.arange(0,101,10),fontsize=14)
		plt.title("Percentage Optimal Action for " + str(self.N) + " Bandits",fontsize=22)
		plt.ylabel("Percentage Optimal Action for " + str(self.runs) + " Runs",fontsize=16)
		plt.xlabel("Time Steps",fontsize=16)
		plt.legend(loc="upper left",fontsize=14)
		plt.savefig("Percentage Optimal Action",bbox_inches="tight")

		plt.show()

	def test(self, agent, time_steps = 1000, runs = 2000):
		self.time_steps = time_steps
		self.runs = runs

		rewards_list = np.zeros(self.time_steps)
		opt_action_list = np.zeros(self.time_steps)
		for i in range(self.runs):
			self.reset()
			agent.reset()		
			
			for j in range(self.time_steps):				
				action = agent.act()
				reward = self.calculate_reward(action)
				agent.update_table(action,reward)
				rewards_list[j] += reward
				if action == self.optimal_action:
					opt_action_list[j] += 1											
		
		opt_action_per_list = opt_action_list / self.runs * 100
		avg_rewards_list = rewards_list / self.runs

		self.plot(avg_rewards_list, opt_action_per_list, agent)			


def experiment1():
	"""
	Four epsilon-greedy agents with epsilon 0, 0.01, 0.1 and 0.5
	are created and compared.
	"""
	agent1 = epsilon_greedy_agent(10, epsilon = 0, initial_value = 0)
	agent2 = epsilon_greedy_agent(10, epsilon = 0.01, initial_value = 0)
	agent3 = epsilon_greedy_agent(10, epsilon = 0.1, initial_value = 0)
	agent4 = epsilon_greedy_agent(10, epsilon = 0.5, initial_value = 0)
	tb = bandit_problem_test_bed(10)
	tb.test(agent1,time_steps = 1000, runs = 2000)
	tb.test(agent2,time_steps = 1000, runs = 2000)
	tb.test(agent3,time_steps = 1000, runs = 2000)
	tb.test(agent4,time_steps = 1000, runs = 2000)
	tb.show_figures()


def experiment2():
	"""
	To demostrate the effect of optimistic initial values
	"""
	agent1 = epsilon_greedy_agent(10, epsilon = 0.0, initial_value = 5)
	agent2 = epsilon_greedy_agent(10, epsilon = 0.1, initial_value = 0)
	tb = bandit_problem_test_bed(10)
	tb.test(agent1,time_steps = 1000, runs = 2000)
	tb.test(agent2,time_steps = 1000, runs = 2000)
	tb.show_figures()


if __name__ == '__main__':
	# experiment1()
	experiment2()