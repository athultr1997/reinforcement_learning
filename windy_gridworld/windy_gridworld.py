import numpy as np
import matplotlib.pyplot as plt
import os
from environments import SimpleEnvironment, WindyEnvironment
from agents import TDAgent

def game1():
	"""
	A TD agent learning in a simple gridworld without any wind.
	"""
	env = SimpleEnvironment()
	agent = TDAgent()
	agent.SARSA(env)

def game2():
	"""
	A TD agent learning in a windy gridworld with 4 actions.
	"""
	env = WindyEnvironment(grid_shape=(7,10), num_actions=4)
	agent = TDAgent()
	agent.SARSA(env)

def game3():
	"""
	A TD agent learning in a windy gridworld with 8 actions.
	"""
	env = WindyEnvironment(grid_shape=(7,10), num_actions=8)
	agent = TDAgent()
	agent.SARSA(env)

		
if __name__ == '__main__':
	# game1()
	# game2()
	game3()