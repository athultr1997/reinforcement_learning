import numpy as np

"""
Definitions
-----------
Actions - {20,...,-20} where 'a' means move 'a' cars from location 1 to location 2
(Negative actions correspond to moving cars from location 2 to location 1)
"""

def poisson_pmf(x, lam):
	return np.power(lam,x) * np.exp(-lam) / np.math.factorial(x)


class agent:
	def __init__(self, max_cars, max_cars_move, poisson_cutoff, rent_lam1, rent_lam2, ret_lam1, ret_lam2, disc_factor, accuracy):
		self.max_cars = max_cars
		self.value_table = np.zeros([max_cars+1, max_cars+1])
		self.policy = np.zeros([max_cars+1, max_cars+1])
		self.actions = np.arange(-max_cars_move, max_cars_move+1)
		self.states = [(i,j) for i in range(self.max_cars + 1) for j in range(self.max_cars + 1)]
		self.num_states = np.power(self.max_cars + 1, 2)
		self.poisson_cutoff = poisson_cutoff
		self.rent_lam1 = rent_lam1
		self.rent_lam2 = rent_lam2
		self.ret_lam1 = ret_lam1
		self.ret_lam2 = ret_lam2
		self.disc_factor = disc_factor
		self.accuracy = accuracy


	def evaluate_policy(self):		
		for i in range(self.max_cars+1):
			for j in range(self.max_cars+1):				
				v = None
				while v == None or np.absolute(v - self.value_table[i][j]) > self.accuracy:
					v = self.value_table[i][j]				
					self.value_table[i][j] = self.find_expected_value(i,j,self.policy[i][j])

		print("policy = ", self.policy)
		print("value_table = ", self.value_table)


	def improve_policy(self):				
		for i in range(10):
			print("iter = ", i)
			policy_stable = True

			for i in range(self.max_cars + 1):
				for j in range(self.max_cars + 1):				
					max_action = None
					max_exp_val = None

					for a in self.actions:
						exp_val = self.find_expected_value(i,j,a)
						if max_exp_val == None or exp_val > max_exp_val:
							max_exp_val = exp_val
							max_action = a

					if max_action != self.policy[i][j]:
						policy[i][j] = max_action
						policy_stable = False					

			if policy_stable == True:
				break
			else:
				self.evaluate_policy()


	def find_expected_value(self, i, j, a):
		exp_val = 0
		
		cars1 = min(i - a, self.max_cars)
		cars2 = min(j + a, self.max_cars)

		for rent1 in range(0, self.poisson_cutoff):
			for rent2 in range(0, self.poisson_cutoff):
				rent_actual1 = min(rent1, cars1)
				rent_actual2 = min(rent2, cars2)

				rewards = 10 * (rent_actual1 + rent_actual2) + (-2) * np.absolute(a)

				rent_prob = poisson_pmf(rent1, self.rent_lam1) * poisson_pmf(rent2, self.rent_lam2)

				for ret1 in range(0, self.poisson_cutoff):
					for ret2 in range(0, self.poisson_cutoff):
						cars1 = min(cars1 - rent_actual1 + ret1, self.max_cars)
						cars2 = min(cars2 - rent_actual2 + ret2, self.max_cars)
						
						ret_prob = poisson_pmf(ret1, self.ret_lam1) * poisson_pmf(ret1, self.ret_lam2)

						prob = rent_prob * ret_prob

						if cars1 >= 0 and cars2 >= 0:						
							exp_val += prob * (rewards + self.disc_factor * self.value_table[int(cars1)][int(cars2)])

		return exp_val	
			

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
	agent = agent(max_cars = 20, max_cars_move = 5, poisson_cutoff = 10, rent_lam1 = 3, rent_lam2 = 4, ret_lam1 = 3, ret_lam2 = 2, disc_factor = 0.9, accuracy = 0.01)
	# agent.evaluate_policy()
	agent.improve_policy()
	# tb = jacks_car_rental_test_bed(rent_lam1 = 3, rent_lam2 = 4, ret_lam1 = 3, ret_lam2 = 2, max_cars = 20)
	print(agent.actions)

