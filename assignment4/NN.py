#!/Library/Frameworks/Python.framework/Versions/3.4/bin/python3
# Joel Doumit
# CS472 - Evolutionary Algorithms
# Assignment 4
# Tree code re-used from Assignment 3.
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer
import numpy as np
import csv

def fitness(NN, data, diags):
	sumRight = 0
	totalPoints= len(data)

	for i in range(totalPoints):
		answer = NN.activate(data[i])
		if answer > 0:
			answer = 1
		if answer < 0 or answer == 0:
			answer = 0
		if answer == diags[i]:
			sumRight += 1

	return sumRight/totalPoints

def selectionPool(population):
	chosenIndiv = np.random.choice(100, 5, replace=False)
	selectPool = []
	for i in chosenIndiv:
		selectPool.append(population[i])
	return selectPool

def sortPool(to_sort):
	to_sort.sort(key=lambda x: x[1], reverse=True)
	return to_sort

def pickParents(sorted_pool):
	return sorted_pool[0][0], sorted_pool[1][0]

def Offspring(parent1, parent2):
	child1 = parent1
	child2 = parent2
	temp = 0
	crossover = np.random.choice(range(1,29)) 
	for i in range(30):
		if i < crossover:
			temp = child1.params[i]
			child1.params[i]=parent2.params[i]
			child2.params[i]= temp

	mutate(child1)
	mutate(child2)

	return child1, child2

def mutate(kid):
	for j in range(30):
		if(np.random.choice([0,1], p=[0.875, 0.125]) == 1):
			kid.params[j] = np.random.uniform(-10, 10)
	return kid

def survivors(sortedPop):
	sortedPop = sortedPop[:len(sortedPop)-2]
	return sortedPop

def avgFit(population):
	average = []
	for i in range(5):
		average.append(population[i][1])
	avg = sum(average)/len(average)
	return avg

def stats(sortedPop):
	best = sortedPop[0][1]
	avg = avgFit(sortedPop)
	return best, avg

if __name__=='__main__':
	with open('Assign4_JSD_NN.csv', mode='w', newline = '') as outfile:
		writer = csv.writer(outfile)
		writer.writerow(['Trial','Generation','Best','Avg',])
		cancer = load_breast_cancer()

		x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

		for t in range(15):

			Population = []

			for i in range(100):
				n = FeedForwardNetwork()

				inLayer = LinearLayer(30)
				hiddenLayer = SigmoidLayer(3)
				outLayer = LinearLayer(1)

				n.addInputModule(inLayer)
				n.addModule(hiddenLayer)
				n.addOutputModule(outLayer)

				in_to_hidden = FullConnection(inLayer, hiddenLayer)
				hidden_to_out = FullConnection(hiddenLayer, outLayer)

				n.addConnection(in_to_hidden)
				n.addConnection(hidden_to_out)

				n.sortModules()

				Population.append([n, fitness(n, x_train, y_train)])


			for g in range(500):
				parent1, parent2 = pickParents(sortPool(selectionPool(Population)))
				child1, child2 = (Offspring(parent1, parent2))
				Population.append([child1, fitness(child1, x_train, y_train)])
				Population.append([child2, fitness(child2, x_train, y_train)])
				sortPool(Population)
				Population = (survivors(Population))
				Best, Avg = stats(Population)
				writer.writerow([t,g,Best,Avg])

			print(Population[0][0].params)