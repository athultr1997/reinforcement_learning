import numpy as np
import matplotlib.pyplot as plt

"""
Definitions
-----------
Actions are indexed from '0' to 'N-1' if there are 'N' actions
"""

class greedy_action_value_agent:
	def __init__(self, N = 10, epsilon = 0.1):
		"""
		table is the DS which stores the values for each actions
		table[i][0] -> The estimate of the value of action 'i'
		table[i][1] -> Sum of rewards when action 'i' was taken prior to time 't'
		table[i][2] -> No:of times action 'i' was taken prior to time 't'
		"""
		self.N = N
		self.epsilon = epsilon
		self.table = list()

		for i in range(self.N):
			self.table.append([0,0,0])

	def take_action(self):
		s = np.random.uniform(low = 0.0, high = 1.0, size = None)

		if s > self.epsilon:
			maxima_action = 0
			for i in range(self.N):
				if self.table[i][0]>self.table[maxima_action][0]:
					maxima_action = i
			return maxima_action
		else:
			l = np.random.uniform(low = 0.0, high = self.N, size = None)
			return int(l)

	def update_table(self,action,reward):
 		self.table[action][1] += reward
 		self.table[action][2] += 1
 		self.table[action][0] = self.table[action][1] / self.table[action][2]

	def reset(self):
 		for i in range(self.N):
 			for j in range(3):
 				self.table[i][j] = 0
		

class bandit_problem_test_bed:
	def __init__(self, N):
		self.N = N
		self.true_values = [None] * self.N

	def reset(self):
		for i in range(self.N):
			self.true_values[i] = np.random.normal(loc = 0.0, scale = 1.0, size = None)
		
		# plt.scatter(range(self.N), self.true_values, marker = 'o', c = 'r')
		# plt.show()

	def calculate_reward(self, action):
		return np.random.normal(loc = self.true_values[action], scale = 1.0, size = None)

	def test(self, agent, time_steps = 1000, runs = 2000):
		self.time_steps = time_steps
		self.runs = runs

		avg_reward = [0] * time_steps
		for i in range(self.runs):
			self.reset()
			agent.reset()
			rewards = list()
			
			for j in range(self.time_steps):
				action = agent.take_action()
				reward = self.calculate_reward(action)
				agent.update_table(action,reward)
				rewards.append(reward)
				# print(agent.table)
			avg_reward = [(((i)*ar)+(r))/(i+1) for ar,r in zip(avg_reward,rewards)]

		plt.plot(range(self.time_steps),avg_reward,label = "$\epsilon$ = " + str(agent.epsilon))
	
	def show_figures(self):
		plt.ylabel("Average Rewards for " + str(self.runs) + " Runs")
		plt.xlabel("Time Step")
		plt.legend()
		plt.show()


def experiment1():
	agent1 = greedy_action_value_agent(10, epsilon = 0)
	agent2 = greedy_action_value_agent(10, epsilon = 0.01)
	agent3 = greedy_action_value_agent(10, epsilon = 0.1)
	agent4 = greedy_action_value_agent(10, epsilon = 0.5)
	tb = bandit_problem_test_bed(10)
	tb.test(agent1,time_steps = 1000, runs = 2000)
	tb.test(agent2,time_steps = 1000, runs = 2000)
	tb.test(agent3,time_steps = 1000, runs = 2000)
	tb.test(agent4,time_steps = 1000, runs = 2000)
	tb.show_figures()


if __name__ == '__main__':
	experiment1()

	




