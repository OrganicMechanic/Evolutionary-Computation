#!/Library/Frameworks/Python.framework/Versions/3.4/bin/python3
# Joel Doumit
# CS472 - Evolutionary Algorithms
# Assignment 2a - Island Model
# Fitness functions used taken from the DEAP Github page,
# https://github.com/DEAP/deap/blob/master/deap/benchmarks/__init__.py

import numpy as np
import csv
import sys
import math
from functools import reduce
from operator import mul

# sys.argv[1] is the way to get commandline arguments.

populationSize = 10
numberOfGenes = 30

def initializeBoard(x, y, z):
	return np.random.uniform(x, y, z)

def initializePopulation(x, y, z, size):
	return [initializeBoard(x, y, z) for i in range(size)]

def PickMigrants(population):
	chosenIndiv = np.random.choice(populationSize, 2, replace=False)
	selectPool = []
	for i in chosenIndiv:
		selectPool.append(population[i])
		del population[i]
	return selectPool

def selectionPool(population):
	chosenIndiv = np.random.choice(populationSize, 5, replace=False)
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
	crossover = np.random.choice(range(1,numberOfGenes-1)) 
	for kid in range(2):
		j = 0
		for i in range(numberOfGenes):
			if i < crossover:
				kids[(0+kid)%2].append(parents[(0+kid)%2][i])
			else:
				kids[(0+kid)%2].append(parents[(1+kid)%2][i])
	return np.asarray(kids)

#### SPHERICAL FUNCTIONS ####
def pairSphereIndividuals(population):
	sphereFitnessPop = []
	for i in population:
		sphereFitnessPop.append([i, sphereFitness(i)])
	return sphereFitnessPop

def sphereFitness(individual):
	return sum(gene * gene for gene in individual)

def sphereMutation(kids):
	#swap two values inside the kid for permutation, pick one
	#value to swap to another random value in combination.
	for i in kids:
		for j in range(numberOfGenes):
			if(np.random.choice([0,1], p=[0.875, 0.125]) == 1):
				i[j] = np.random.uniform(-5.12, 5.12)
	return kids

def sphereGeneration(population):
	newPopulation = []
	parents = pickParents(sortPool(selectionPool(population)))
	kids = (crossover(parents))
	sphereMutation(kids)
	newPopulation.append([kids[0], sphereFitness(kids[0])])
	newPopulation.append([kids[1], sphereFitness(kids[1])])
	return newPopulation

#### ROSENBROCK FUNCTIONS ####
def pairRosenIndividuals(population):
	rosenFitnessPop = []
	for i in population:
		rosenFitnessPop.append([i, rosenFitness(i)])
	return rosenFitnessPop

def rosenFitness(individual):
	return sum(100 * (x * x - y)**2 + (1. - x)**2 \
for x, y in zip(individual[:-1], individual[1:]))

def rosenMutation(kids):
	#swap two values inside the kid for permutation, pick one
	#value to swap to another random value in combination.
	for i in kids:
		for j in range(numberOfGenes):
			if(np.random.choice([0,1], p=[0.875, 0.125]) == 1):
				i[j] = np.random.uniform(-2.048, 2.048)
	return kids

def rosenGeneration(population):
	newPopulation = []
	parents = pickParents(sortPool(selectionPool(population)))
	kids = (crossover(parents))
	rosenMutation(kids)
	newPopulation.append([kids[0], rosenFitness(kids[0])])
	newPopulation.append([kids[1], rosenFitness(kids[1])])
	return newPopulation

#### RASTRIGIN FUNCTIONS ####
def pairRastIndividuals(population):
	rastFitnessPop = []
	for i in population:
		rastFitnessPop.append([i, rastFitness(i)])
	return rastFitnessPop

def rastFitness(individual):
	return 10 * len(individual) + sum(gene * gene - 10 * \
math.cos(2 * math.pi * gene) for gene in individual)

def rastMutation(kids):
	for i in kids:
		for j in range(numberOfGenes):
			if(np.random.choice([0,1], p=[0.875, 0.125]) == 1):
				i[j] = np.random.uniform(-5.12, 5.12)
	return kids

def rastGeneration(population):
	newPopulation = []
	parents = pickParents(sortPool(selectionPool(population)))
	kids = (crossover(parents))
	rastMutation(kids)
	newPopulation.append([kids[0], rastFitness(kids[0])])
	newPopulation.append([kids[1], rastFitness(kids[1])])
	return newPopulation

### SCHWEFEL FUNCTIONS ###
def pairSchwefelIndividuals(population):
	schwefelFitnessPop = []
	for i in population:
		schwefelFitnessPop.append([i, schwefelFitness(i)])
	return schwefelFitnessPop

def schwefelFitness(individual):
	return (418.9828872724339*30-sum(x*math.sin(math.sqrt(np.abs(x))) for x in individual))

def schwefelMutation(kids):
	for i in kids:
		for j in range(numberOfGenes):
			if(np.random.choice([0,1], p=[0.875, 0.125]) == 1):
				i[j] = np.random.uniform(-30, 30)
	return kids

def schwefelGeneration(population):
	newPopulation = []
	parents = pickParents(sortPool(selectionPool(population)))
	kids = (crossover(parents))
	schwefelMutation(kids)
	newPopulation.append([kids[0], schwefelFitness(kids[0])])
	newPopulation.append([kids[1], schwefelFitness(kids[1])])
	return newPopulation

### ACKLEY FUNCTIONS ### 
def pairAckleyIndividuals(population):
	ackleyFitnessPop = []
	for i in population:
		ackleyFitnessPop.append([i, ackleyFitness(i)])
	return ackleyFitnessPop

def ackleyFitness(individual):
	return 20 - 20 * np.exp(-0.2*math.sqrt(1.0/30 * sum(x**2 for x in individual))) \
+ np.e - np.exp(1.0/30 * sum(math.cos(2*math.pi*x) for x in individual))

def ackleyMutation(kids):
	for i in kids:
		for j in range(numberOfGenes):
			if(np.random.choice([0,1], p=[0.875, 0.125]) == 1):
				i[j] = np.random.uniform(-30, 30)
	return kids

def ackleyGeneration(population):
	newPopulation = []
	parents = pickParents(sortPool(selectionPool(population)))
	kids = (crossover(parents))
	ackleyMutation(kids)
	newPopulation.append([kids[0], ackleyFitness(kids[0])])
	newPopulation.append([kids[1], ackleyFitness(kids[1])])
	return newPopulation

### GRIEWANGK FUNCTIONS ###
def pairGriewangkIndividuals(population):
	griewangkFitnessPop = []
	for i in population:
		griewangkFitnessPop.append([i, griewangkFitness(i)])
	return griewangkFitnessPop

def griewangkFitness(individual):
	return 1.0/4000.0 * sum(x**2 for x in individual) - \
reduce(mul, (math.cos(x/math.sqrt(i+1.0)) for i, x in enumerate(individual)), 1) + 1

def griewangkMutation(kids):
	for i in kids:
		for j in range(numberOfGenes):
			if(np.random.choice([0,1], p=[0.875, 0.125]) == 1):
				i[j] = np.random.uniform(-600, 600)
	return kids

def griewangkGeneration(population):
	newPopulation = []
	parents = pickParents(sortPool(selectionPool(population)))
	kids = (crossover(parents))
	griewangkMutation(kids)
	newPopulation.append([kids[0], griewangkFitness(kids[0])])
	newPopulation.append([kids[1], griewangkFitness(kids[1])])
	return newPopulation

if __name__ == "__main__":
	method = sys.argv[1]

	if method == "spherical":
		newSpherePop1 = []
		newSpherePop2 = []
		newSpherePop3 = []

		initPop1 = initializePopulation(-5.12, 5.12, numberOfGenes, populationSize)
		initPop2 = initializePopulation(-5.12, 5.12, numberOfGenes, populationSize)
		initPop3 = initializePopulation(-5.12, 5.12, numberOfGenes, populationSize)
		spherePopulation1 = pairSphereIndividuals(initPop1)
		spherePopulation2 = pairSphereIndividuals(initPop2)
		spherePopulation3 = pairSphereIndividuals(initPop3)

		print(spherePopulation1)

		for i in range(50):
			for g in range(5): #the range should be the population size/2
				newSpherePop1.extend(sphereGeneration(spherePopulation1))

			for g in range(5): #the range should be the population size/2
				newSpherePop2.extend(sphereGeneration(spherePopulation2))

			for g in range(5): #the range should be the population size/2
				newSpherePop3.extend(sphereGeneration(spherePopulation3))

			spherePopulation1 = newSpherePop1
			spherePopulation2 = newSpherePop2
			spherePopulation3 = newSpherePop3
			newSpherePop1 = []
			newSpherePop2 = []
			newSpherePop3 = []

		pop1Migrants = PickMigrants(spherePopulation1)
		pop2Migrants = PickMigrants(spherePopulation2)
		pop3Migrants = PickMigrants(spherePopulation3)

		spherePopulation2.extend(pop1Migrants)
		spherePopulation3.extend(pop2Migrants)
		spherePopulation1.extend(pop3Migrants)

	elif method == "rosenbrock":
		newRosenPop1 = []
		newRosenPop2 = []
		newRosenPop3 = []

		initPop1 = initializePopulation(-2.048, 2.048, numberOfGenes, populationSize)
		initPop2 = initializePopulation(-2.048, 2.048, numberOfGenes, populationSize)
		initPop3 = initializePopulation(-2.048, 2.048, numberOfGenes, populationSize)
		rosenPopulation1 = pairRosenIndividuals(initPop1)
		rosenPopulation2 = pairRosenIndividuals(initPop2)
		rosenPopulation3 = pairRosenIndividuals(initPop3)

		print(rosenPopulation1)

		for i in range(50):
			for g in range(5): #the range should be the population size/2
				newRosenPop1.extend(rosenGeneration(rosenPopulation1))

			for g in range(5): #the range should be the population size/2
				newRosenPop2.extend(rosenGeneration(rosenPopulation2))

			for g in range(5): #the range should be the population size/2
				newRosenPop3.extend(rosenGeneration(rosenPopulation3))

			rosenPopulation1 = newRosenPop1
			rosenPopulation2 = newRosenPop2
			rosenPopulation3 = newRosenPop3
			newRosenPop1 = []
			newRosenPop2 = []
			newRosenPop3 = []

		pop1Migrants = PickMigrants(rosenPopulation1)
		pop2Migrants = PickMigrants(rosenPopulation2)
		pop3Migrants = PickMigrants(rosenPopulation3)

		rosenPopulation2.extend(pop1Migrants)
		rosenPopulation3.extend(pop2Migrants)
		rosenPopulation1.extend(pop3Migrants)

	elif method == "rastrigin":
		newRastPop1 = []
		newRastPop2 = []
		newRastPop3 = []
		
		initPop1 = initializePopulation(-5.12, 5.12, numberOfGenes, populationSize)
		initPop2 = initializePopulation(-5.12, 5.12, numberOfGenes, populationSize)
		initPop3 = initializePopulation(-5.12, 5.12, numberOfGenes, populationSize)
		rastPopulation1 = pairRastIndividuals(initPop1)
		rastPopulation2 = pairRastIndividuals(initPop2)
		rastPopulation3 = pairRastIndividuals(initPop3)

		print(rastPopulation1)

		for i in range(50):
			for g in range(5): #the range should be the population size/2
				newRastPop1.extend(rastGeneration(rastPopulation1))

			for g in range(5): #the range should be the population size/2
				newRastPop2.extend(rastGeneration(rastPopulation2))

			for g in range(5): #the range should be the population size/2
				newRastPop3.extend(rastGeneration(rastPopulation3))

			rastPopulation1 = newRastPop1
			rastPopulation2 = newRastPop2
			rastPopulation3 = newRastPop3
			newRastPop1 = []
			newRastPop2 = []
			newRastPop3 = []

		pop1Migrants = PickMigrants(rastPopulation1)
		pop2Migrants = PickMigrants(rastPopulation2)
		pop3Migrants = PickMigrants(rastPopulation3)

		print ("Migrants from population 1:")
		print(pop1Migrants)

		rastPopulation2.extend(pop1Migrants)
		rastPopulation3.extend(pop2Migrants)
		rastPopulation1.extend(pop3Migrants)

	elif method == "schwefel":
		newSchwefelPop1 = []
		newSchwefelPop2 = []
		newSchwefelPop3 = []
		
		initPop1 = initializePopulation(-512.03, 511.97, numberOfGenes, populationSize)
		initPop2 = initializePopulation(-512.03, 511.97, numberOfGenes, populationSize)
		initPop3 = initializePopulation(-512.03, 511.97, numberOfGenes, populationSize)
		schwefelPopulation1 = pairSchwefelIndividuals(initPop1)
		schwefelPopulation2 = pairSchwefelIndividuals(initPop2)
		schwefelPopulation3 = pairSchwefelIndividuals(initPop3)

		print(schwefelPopulation1)

		for i in range(50):
			for g in range(5): #the range should be the population size/2
				newSchwefelPop1.extend(schwefelGeneration(schwefelPopulation1))

			for g in range(5): #the range should be the population size/2
				newSchwefelPop2.extend(schwefelGeneration(schwefelPopulation2))

			for g in range(5): #the range should be the population size/2
				newSchwefelPop3.extend(schwefelGeneration(schwefelPopulation3))

			schwefelPopulation1 = newSchwefelPop1
			schwefelPopulation2 = newSchwefelPop2
			schwefelPopulation3 = newSchwefelPop3
			newSchwefelPop1 = []
			newSchwefelPop2 = []
			newSchwefelPop3 = []

		pop1Migrants = PickMigrants(schwefelPopulation1)
		pop2Migrants = PickMigrants(schwefelPopulation2)
		pop3Migrants = PickMigrants(schwefelPopulation3)

		schwefelPopulation2.extend(pop1Migrants)
		schwefelPopulation3.extend(pop2Migrants)
		schwefelPopulation1.extend(pop3Migrants)

	elif method == "ackley":
		newAckleyPop1 = []
		newAckleyPop2 = []
		newAckleyPop3 = []
		
		initPop1 = initializePopulation(-30, 30, numberOfGenes, populationSize)
		initPop2 = initializePopulation(-30, 30, numberOfGenes, populationSize)
		initPop3 = initializePopulation(-30, 30, numberOfGenes, populationSize)
		ackleyPopulation1 = pairAckleyIndividuals(initPop1)
		ackleyPopulation2 = pairAckleyIndividuals(initPop2)
		ackleyPopulation3 = pairAckleyIndividuals(initPop3)

		print(ackleyPopulation1)

		for i in range(50):
			for g in range(5): #the range should be the population size/2
				newAckleyPop1.extend(ackleyGeneration(ackleyPopulation1))

			for g in range(5): #the range should be the population size/2
				newAckleyPop2.extend(ackleyGeneration(ackleyPopulation2))

			for g in range(5): #the range should be the population size/2
				newAckleyPop3.extend(ackleyGeneration(ackleyPopulation3))

			ackleyPopulation1 = newAckleyPop1
			ackleyPopulation2 = newAckleyPop2
			ackleyPopulation3 = newAckleyPop3
			newAckleyPop1 = []
			newAckleyPop2 = []
			newAckleyPop3 = []

		pop1Migrants = PickMigrants(ackleyPopulation1)
		pop2Migrants = PickMigrants(ackleyPopulation2)
		pop3Migrants = PickMigrants(ackleyPopulation3)

		ackleyPopulation2.extend(pop1Migrants)
		ackleyPopulation3.extend(pop2Migrants)
		ackleyPopulation1.extend(pop3Migrants)

	elif method == "griewangk":
		newGriewangkPop1 = []
		newGriewangkPop2 = []
		newGriewangkPop3 = []
		
		initPop1 = initializePopulation(-600, 600, numberOfGenes, populationSize)
		initPop2 = initializePopulation(-600, 600, numberOfGenes, populationSize)
		initPop3 = initializePopulation(-600, 600, numberOfGenes, populationSize)
		griewangkPopulation1 = pairGriewangkIndividuals(initPop1)
		griewangkPopulation2 = pairGriewangkIndividuals(initPop2)
		griewangkPopulation3 = pairGriewangkIndividuals(initPop3)

		print(griewangkPopulation1)

		for i in range(50):
			for g in range(5): #the range should be the population size/2
				newGriewangkPop1.extend(griewangkGeneration(griewangkPopulation1))

			for g in range(5): #the range should be the population size/2
				newGriewangkPop2.extend(griewangkGeneration(griewangkPopulation2))

			for g in range(5): #the range should be the population size/2
				newGriewangkPop3.extend(griewangkGeneration(griewangkPopulation3))

			griewangkPopulation1 = newGriewangkPop1
			griewangkPopulation2 = newGriewangkPop2
			griewangkPopulation3 = newGriewangkPop3
			newGriewangkPop1 = []
			newGriewangkPop2 = []
			newGriewangkPop3 = []

		pop1Migrants = PickMigrants(griewangkPopulation1)
		pop2Migrants = PickMigrants(griewangkPopulation2)
		pop3Migrants = PickMigrants(griewangkPopulation3)

		griewangkPopulation2.extend(pop1Migrants)
		griewangkPopulation3.extend(pop2Migrants)
		griewangkPopulation1.extend(pop3Migrants)

	else:
		print ("DING DONG YOUR ARGUMENT IS WRONG")
