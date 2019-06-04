#!/Library/Frameworks/Python.framework/Versions/3.4/bin/python3
# Joel Doumit
# CS472 - Evolutionary Algorithms
# Assignment 3 - Symbolic Regression
# NOTE: several functions (for sorting, selection, survivorship and stats) are
# re-used from my assignment 1. Additionally, some code of the tree code used
# came from Nikhil Kumar Singh, on the Geeksforgeeks website.
# https://www.geeksforgeeks.org/write-a-c-program-to-find-the-maximum-depth-or-height-of-a-tree/

import numpy as np
import random
import csv

operators = ["+", "-", "*", "/", "sqrt"]
terminals = ["x", -10, -5, -4, -3, -2, 0, -1, 1, 2, 3, 4, 5, 10]

class node: 
	def __init__(self, value): 
		self.left = None
		self.data = value 
		self.right = None

def evaluateTree(root, x): 
	# empty tree 
	if root is None: 
		return 0

	if root.data == "x":
		return x

	# leaf node 
	if root.left is None and root.right is None: 
		return int(root.data) 

	# evaluate left tree 
	left_sum = evaluateTree(root.left, x) 

	# evaluate right tree 
	right_sum = evaluateTree(root.right, x) 

	if root.data == '+': 
		return left_sum + right_sum

	elif root.data == '-': 
		return left_sum - right_sum

	elif root.data == '*': 
		return left_sum * right_sum

	elif root.data == 'sqrt':
		if right_sum >= 0:
			return np.sqrt(right_sum)
		else:
			return np.sqrt(-1*right_sum)

	else: 
		if(right_sum == 0):
			return 0
		else:
			return left_sum / right_sum 

def printTree(root):
	if root is None: 
		print("Nothing")
	else:
		if root.left is None and root.right is None:
			print(root.data, end='')
		else:
			if root.data == 'sqrt':
				print("(", end='')
				print(root.data, end='')
				printTree(root.right)
				print(")", end='')
			else:
				print("(", end='')
				printTree(root.left)
				print(root.data, end='')
				printTree(root.right)
				print(")", end='')

def nodeVal(operChanceLow, operChanceHigh):
	if (np.random.choice([0,1], p=[operChanceLow, operChanceHigh])):
		return random.choice(operators)
	else:
		return random.choice(terminals)


def generateTerminal():
	return random.choice(terminals)

def generateFull(depth=1):
	if depth < 4:
		root = node(nodeVal(0, 1))
		root.left = generateFull(depth+1)
		root.right = generateFull(depth+1)
		return root
	else:
		root = node(nodeVal(1,0))
		return root

def generateGrow(depth=1):
	if depth < 4:
		root = node(nodeVal(0.1, 0.9))
		if root.data in operators:
			root.left = generateGrow(depth+1)
			root.right = generateGrow(depth+1)
		return root
	else:
		root = node(nodeVal(1,0))
		return root

def Copy(source):
	if(source):
		new = node(source.data)
		new.data = source.data
		if source.left:
			new.left = (Copy(source.left))
		if source.right:
			new.right = (Copy(source.right))
	return new

def Delete(population, index):
	del population[index]

def keyFunction(x):
	return(-0.11*(x**5)+(x**3)+8*x)

def fitness(tree, keyAnswers, usePM, population):
	error = []
	answerArray = []
	sumErr = 0

	for i in range(5):
		answerArray.append(evaluateTree(tree, i+1))

	for i in range(5):
		error.append(answerArray[i] - keyAnswers[i])

	for i in error:
		sumErr += (i**2)
		mean = sumErr/len(error)
		RMSE = np.sqrt(mean)
		if usePM:
			if (maxDepth(tree)>6):
				parsimony = avgFit(population)
				fitness = RMSE + parsimony
				return fitness
		else:
			return RMSE

def avgFit(population):
	average = []
	for i in range(100):
		average.append(population[i][1])
	avg = sum(average)/len(average)
	return avg

def maxDepth(node):
	if node is None:
		return 0 ;
	else :
		# Compute the depth of each subtree
		lDepth = maxDepth(node.left)
		rDepth = maxDepth(node.right)
		# Use the larger one
		if (lDepth > rDepth):
			return lDepth+1
		else:
			return rDepth+1

def Offspring(parent1, parent2):
	child1 = Copy(parent1)
	child2 = Copy(parent2)
	n1 = np.random.choice(Nodelist(child1))
	n2 = np.random.choice(Nodelist(child2))
	n3 = node(None)
	n3.data = n1.data
	n3.left = n1.left
	n3.right = n1.right
	n1.data = n2.data
	n1.left = n2.left
	n1.right = n2.right
	n2.data = n3.data
	n2.left = n3.left
	n2.right = n3.right

	mutation(child1)
	mutation(child2)
	return child1, child2

def Nodelist(root):
	if(root):
		list = []
		list.append(root)
		if((root.left) and (root.right)):
			for l in Nodelist(root.left):
				list.append(l)
			for l in Nodelist(root.right):
				list.append(l)
		if(root.left):
			for l in Nodelist(root.left):
				list.append(l)
		if(root.right):
			for l in Nodelist(root.right):
				list.append(l)
	return list

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

def mutation(kid):
	n1 = np.random.choice(Nodelist(kid))
	if n1.data in operators:
		n1.data = random.choice(operators)
	if n1.data in terminals:
		n1.data = random.choice(terminals)


def survivors(sortedPop):
	sortedPop = sortedPop[:len(sortedPop)-2]
	return sortedPop

def stats(sortedPop):
	best = sortedPop[0][1]
	avg = avgFit(sortedPop)
	return best, avg

###################################################################
############### MAIN ##############################################
###################################################################

if __name__=='__main__': 
	with open('Assign3_JSD.csv', mode='w', newline = '') as outfile:
		writer = csv.writer(outfile)
		writer.writerow(['Generation','Best','Avg'])

		bestTrees =[]
		for k in range(15):
			keyAnswers =[]
			for i in range(5):
				keyAnswers.append(keyFunction(i+1))

		# Initialization
			Population = []
			for i in range(50):
				tree = generateFull()
				Population.append([tree, fitness(tree, keyAnswers, 0, Population)])

			for i in range(50):
				tree = generateGrow()
				Population.append([tree, fitness(tree, keyAnswers, 0, Population)])

		# Selection
			for g in range(500):
				parents = pickParents(sortPool(selectionPool(Population)))
				child1, child2 = Offspring(parents[0], parents[1])
				Population.append([child1, fitness(child1, keyAnswers, 0, Population)])
				Population.append([child2, fitness(child2, keyAnswers, 0, Population)])
				sortPool(Population)
				Population = (survivors(Population))
				best, avg = stats(Population)
				writer.writerow([k,g,best,avg])
			bestTrees.append([Population[0][0], Population[0][1]])
		sortPool(bestTrees)
		printTree((bestTrees[0][0]))
