#!/Library/Frameworks/Python.framework/Versions/3.4/bin/python3
# Joel Doumit
# CS472 - Evolutionary Algorithms
# Assignment 1a - 8 Queens

import numpy as np

def initializeBoard():
	return np.random.permutation(8)

def initializePopulation(size):
	return [initializeBoard() for i in range(size)]

def checkMatch(pos):
	counts = (np.unique(pos, return_counts=True))[1]
	conflicts = [i * (i-1) for i in counts]
	return np.sum(conflicts)

def fitness(board):
	score = 0
	posdiag = [0, 1, 2, 3, 4, 5, 6, 7]
	negdiag = [0, -1, -2, -3, -4, -5, -6, -7]

	checkpos = [x + y for x, y in zip(board, posdiag)]
	checkneg = [x + y for x, y in zip(board, negdiag)]

	score = (checkMatch(checkpos) + checkMatch(checkneg))

	return score

if __name__ == "__main__":
	population = initializePopulation(100)
	for i in population:
		print(i, " - ", fitness(i))
