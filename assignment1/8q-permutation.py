#!/Library/Frameworks/Python.framework/Versions/3.4/bin/python3
# Joel Doumit
# CS472 - Evolutionary Algorithms
# Assignment 1 - 8 Queens - Permutation

import numpy as np
import csv

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

def pairIndividuals(population):
	fitnessPop = []
	for i in population:
		fitnessPop.append([i, fitness(i)])
	return fitnessPop

def selectionPool(population):
	chosenIndiv = np.random.choice(100, 5, replace=False)
	selectPool = []
	for i in chosenIndiv:
		selectPool.append(population[i])
	return selectPool

def sortPool(to_sort):
	to_sort.sort(key=lambda x: x[1])
	return to_sort

def pickParents(sorted_pool):
	return sorted_pool[0][0], sorted_pool[1][0]

def crossover(parents):
	kids = [[],[]]
	crossover = np.random.choice(range(1,7))
	for kid in range(2):
		offset = 0
		for i in range(8):
			if i < crossover:
				kids[(0+kid)%2].append(parents[(0+kid)%2][i]) 
			else:
				while(parents[(1+kid)%2][(i+offset)%8] in kids[(0+kid)%2]):
					offset += 1
				kids[(0+kid)%2].append(parents[(1+kid)%2][(i+offset)%8])
	return np.asarray(kids)

def mutation(kids):
	#swap two values inside the kid for permutation, pick one
	#value to swap to another random value in combination.
	swapVals = np.random.choice(7, 2, replace=False)
	kids[0][swapVals[0]], kids[0][swapVals[1]] = kids[0][swapVals[1]], kids[0][swapVals[0]]
	swapVals = np.random.choice(7, 2, replace=False)
	kids[1][swapVals[0]], kids[1][swapVals[1]] = kids[1][swapVals[1]], kids[1][swapVals[0]]
	return kids

def survivors(sortedPop):
	sortedPop = sortedPop[:len(sortedPop)-2]
	return sortedPop

def stats(sortedPop):
	best = sortedPop[0][1]
	worst = sortedPop[99][1]
	avg = 0
	for i in sortedPop:
		avg = avg + i[1]
	avg = avg / 100
	return best, worst, avg

if __name__ == "__main__":
	with open('8queens_permutation_JSD.csv', mode='w', newline = '') as outfile:
		writer = csv.writer(outfile)
		writer.writerow(['Trial','Generation','Best','Avg','Worst'])

		for t in range(30):
			initPop = initializePopulation(100)
			population = pairIndividuals(initPop)
			for g in range(1000):
				parents = pickParents(sortPool(selectionPool(population)))
				kids = (crossover(parents))
				mutation(kids)
				population.append([kids[0], fitness(kids[0])])
				population.append([kids[1], fitness(kids[1])])
				sortPool(population)
				population = (survivors(population))
				Best, Worst, Avg = stats(population)
				writer.writerow([t,g,Best,Avg,Worst])