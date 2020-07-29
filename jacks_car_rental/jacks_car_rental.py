import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

"""
Definitions
-----------
Actions - {20,...,-20} where 'a' means move 'a' cars from location 1 to location 2
(Negative actions correspond to moving cars from location 2 to location 1)
"""

class Poisson():
	def __init__(self, lam):
		self.lam = lam
		self.sum = 0
		self.epsilon = 0.01
		self.values = {}

		i = self.lam
		while i>=0:
			tmp = poisson.pmf(i, self.lam)
			if tmp > self.epsilon:
				self.values[i] = tmp
				i -= 1
			else:
				break

		self.alpha = i + 1

		i = self.lam + 1
		while True:
			tmp = poisson.pmf(i, self.lam)
			if tmp > self.epsilon:
				self.values[i] = tmp
				i += 1
			else:
				break

		self.beta = i	


class agent:
	def __init__(self, max_cars, max_cars_move, poisson_cutoff, rent_lam1, rent_lam2, ret_lam1, ret_lam2, disc_factor, accuracy):
		self.max_cars = max_cars
		self.max_cars_move = max_cars_move
		self.value_table = np.zeros([self.max_cars+1, self.max_cars+1])
		# self.value_table = np.random.rand(max_cars+1, max_cars+1)
		# self.policy = np.random.randint(low=-max_cars_move,high=max_cars_move+1,size=(max_cars+1, max_cars+1))
		self.policy = np.zeros([self.max_cars+1, self.max_cars+1])
		self.actions = np.arange(-max_cars_move, max_cars_move+1)
		self.states = [(i,j) for i in range(self.max_cars + 1) for j in range(self.max_cars + 1)]
		self.num_states = np.power(self.max_cars + 1, 2)		
		self.rent_lam1 = rent_lam1
		self.rent_lam2 = rent_lam2
		self.ret_lam1 = ret_lam1
		self.ret_lam2 = ret_lam2
		self.disc_factor = disc_factor
		self.accuracy = accuracy

		self.rent_poisson1 = Poisson(self.rent_lam1)
		self.rent_poisson2 = Poisson(self.rent_lam2)
		self.ret_poisson1 = Poisson(self.ret_lam1)
		self.ret_poisson2 = Poisson(self.ret_lam2)


	def evaluate_policy(self):		
		while True:
			delta = 0
			for i in range(self.max_cars+1):
				for j in range(self.max_cars+1):
					old_value = self.value_table[i][j]
					self.value_table[i][j] = self.find_expected_value(i,j,self.policy[i][j])
					delta = np.maximum(delta, np.absolute(old_value - self.value_table[i][j]))

			if delta < self.accuracy:
				break	


	def improve_policy(self):				
		policy_stable = True

		for i in range(self.max_cars + 1):
			for j in range(self.max_cars + 1):				
				max_action = None
				max_exp_val = None

				# max cap is not considered since it can be returned to company	
				min_valid_action = max(-self.max_cars_move, -j)
				max_valid_action = min(self.max_cars_move, i)

				for a in range(min_valid_action,max_valid_action+1):
					exp_val = self.find_expected_value(i,j,a)					
					if max_exp_val == None or exp_val > max_exp_val:
						max_exp_val = exp_val
						max_action = a

				if max_action != self.policy[i][j]:
					self.policy[i][j] = max_action
					policy_stable = False							

		return policy_stable


	def iterate_policy(self):
		i = 0
		
		self.plot(data="policy", figname="policy" + str(i))
		self.plot(data="values", figname="values" + str(i))
		i += 1

		self.evaluate_policy()
		
		# self.plot(data="policy", figname="policy" + str(i))
		# self.plot(data="values", figname="values" + str(i))
		# i += 1

		while self.improve_policy()==False:
			self.plot(data="policy", figname="policy" + str(i))
			self.plot(data="values", figname="values" + str(i))
			i += 1			
			
			self.evaluate_policy()
			
			# self.plot(data="policy", figname="policy" + str(i))
			# self.plot(data="values", figname="values" + str(i))
			# i += 1


	def find_expected_value(self, i, j, a):
		# cars returned today can be rented only tommorrow
		# hence rent1, rent2, ret1, ret2 oreder		
		exp_val = (-2) * np.absolute(a)
		
		cars1 = max(min(i - a, self.max_cars), 0)
		cars2 = max(min(j + a, self.max_cars), 0)		
		
		for rent1 in range(self.rent_poisson1.alpha, self.rent_poisson1.beta):
			for rent2 in range(self.rent_poisson2.alpha, self.rent_poisson2.beta):
				rent_actual1 = min(rent1, cars1)
				rent_actual2 = min(rent2, cars2)

				rewards = 10 * (rent_actual1 + rent_actual2)

				rent_prob = self.rent_poisson1.values[rent1] * self.rent_poisson2.values[rent2]

				for ret1 in range(self.ret_poisson1.alpha, self.ret_poisson1.beta):
					for ret2 in range(self.ret_poisson2.alpha, self.ret_poisson2.beta):
						new_cars1 = max(min(cars1 - rent_actual1 + ret1, self.max_cars),0)
						new_cars2 = max(min(cars2 - rent_actual2 + ret2, self.max_cars),0)
						
						ret_prob = self.ret_poisson1.values[ret1] * self.ret_poisson2.values[ret2]

						prob = rent_prob * ret_prob				

						exp_val += prob * (rewards + self.disc_factor * self.value_table[int(new_cars1)][int(new_cars2)])
						
		return exp_val	
			

	def plot(self, data, figname):
		fig, ax = plt.subplots()
		fig.set_size_inches(10, 10)

		im = None
		if data=="policy":
			im = ax.imshow(self.policy)
		elif data=="values":
			im = ax.imshow(self.value_table)

		cbar = ax.figure.colorbar(im, ax=ax)

		if data=="policy":
			cbar.ax.set_ylabel("Actions", rotation=-90, va="bottom")
		elif data=="values":
			cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")
		
		ax.set_xticks(np.arange(0,self.max_cars+1))
		ax.set_yticks(np.arange(0,self.max_cars+1))
		ax.set_xticklabels(np.arange(0,self.max_cars+1), fontsize=14)
		ax.set_yticklabels(np.arange(0,self.max_cars+1), fontsize=14)
		ax.set_ylabel("Cars at Location 1", fontsize=16)
		ax.set_xlabel("Cars at Location 2", fontsize=16)

		for edge, spine in ax.spines.items():
			spine.set_visible(False)

		ax.set_xticks(np.arange(0,self.max_cars+2)-.5, minor=True)
		ax.set_yticks(np.arange(0,self.max_cars+2)-.5, minor=True)
		ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
		ax.tick_params(which="minor", bottom=False, left=False)

		if data=="policy":
			for i in range(self.max_cars+1):
				for j in range(self.max_cars+1):
					ax.text(j, i, self.policy[i][j],ha="center", va="center", color="w")
		elif data=="values":
			pass
			# for i in range(self.max_cars+1):
			# 	for j in range(self.max_cars+1):
			# 		ax.text(j, i, self.value_table[i][j],ha="center", va="center", color="w")

		if data=="policy":
			ax.set_title("Policy", fontsize=22)
		elif data=="values":
			ax.set_title("Values", fontsize=22)

		fig.tight_layout()
		plt.savefig(figname)


	def act():
		pass


class jacks_car_rental_test_bed:
	def __init__(self, rent_lam1, rent_lam2, ret_lam1, ret_lam2, max_cars):
		self.rent_lam1 = rent_lam1
		self.rent_lam2 = rent_lam2
		self.ret_lam1 = ret_lam1
		self.ret_lam2 = ret_lam2

	def next_time_step():
		cars_rent1 = np.random.poisson(self.rent_lam1)
		cars_rent2 = np.random.poisson(self.rent_lam2)
		cars_ret1 = np.random.poisson(self.ret_lam1)
		cars_ret2 = np.random.poisson(self.ret_lam2)		

		return cars_rent1, cars_rent2, cars_ret1, cars_ret2

	def test(agent):
		pass


if __name__ == '__main__':
	agent = agent(max_cars = 20, max_cars_move = 5, poisson_cutoff = 5, rent_lam1 = 3, rent_lam2 = 4, 
		ret_lam1 = 3, ret_lam2 = 2, disc_factor = 0.9, accuracy = 0.01)
	agent.iterate_policy()
	# print('agent.rent_poisson1.alpha=', agent.rent_poisson1.alpha)
	# print('agent.rent_poisson1.beta=',agent.rent_poisson1.beta)
	# print('agent.rent_poisson2.alpha=',agent.rent_poisson2.alpha)
	# print('agent.rent_poisson2.beta=',agent.rent_poisson2.beta)
	# print('agent.ret_poisson1.alph=',agent.ret_poisson1.alpha)
	# print('agent.ret_poisson1.beta=',agent.ret_poisson1.beta)
	# print('agent.ret_poisson2.alpha=',agent.ret_poisson2.alpha)
	# print('agent.ret_poisson2.beta=',agent.ret_poisson2.beta)
	# print(agent.find_expected_value(10,10,-1))
	# print(agent.find_expected_value(1,1,-1))

	# p = Poisson(4)
	# print(p.alpha)
	# print(p.beta)
	# print(p.values)

	
	# tb = jacks_car_rental_test_bed(rent_lam1 = 3, rent_lam2 = 4, ret_lam1 = 3, ret_lam2 = 2, max_cars = 20)

