
import sys
import nltk
import string
import copy
import math
import random


def tokenize(phrase):
	words = []
	for line in phrase.split("\n"):
		for word in line.split(" "):
			words.append(word.strip(string.punctuation).lower())
	return words
	
def printMatrix(table):
	col_width = [max(len(str(x)) for x in col) for col in zip(*table)]
	for line in table:
		print(" | " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")
	print()	


# Gets the dot product between two vectors
def dot(vector1, vector2):
	if len(vector1) != len(vector2):
		return 0.0
	return sum(i[0] * i[1] for i in zip(vector1, vector2))
	

# Gets the magnitude of a vector (double bars)	
def magnitude(vector):
	return math.sqrt(sum(i**2 for i in vector))
	

# Returns a similarity matrix based on Cosine Similarity
def similarityMatrix(matrix):
	similarity_matrix = [[0] * len(matrix) for i in range(len(matrix))]
	for i in range(len(similarity_matrix)):
		for j in range(len(similarity_matrix[i])):
			similarity_matrix[i][j] = round(1.0 * dot(matrix[i], matrix[j]) / (magnitude(matrix[i]) * magnitude(matrix[j])), 5)
	return similarity_matrix
	
	
def textRank(damper, matrix, iterations):
	# Get probability of going from a node to its neighbor based on matrix
	probability = copy.deepcopy(matrix)
	for i in range(len(matrix)):
		total = sum(x if x != i else 0.0 for x in range(len(matrix[i])))
		for j in range(len(matrix[i])):
			probability[i][j] = 0.0 if i == j else matrix[i][j] / total
	# 0.0 initial score for each sentence
	page_rank = [0] * len(matrix)
	# start walking at a random node
	start_node = random.randint(0, len(page_rank) - 1)
	iteration = 0
	while iteration < iterations:
		rand = random.uniform(0.0, 1.0)
		if rand <= damper:
			# go to neighbor node randomly based on probability
			prob = random.uniform(0.0, 1.0)
			for i in range(len(page_rank)):
				prob -= probability[start_node][i]
				if prob <= 0:
					start_node = i
					break
			page_rank[start_node] += 1
		else:
			# start a new walk at a random node. Only increase iteration when a new walk is started
			iteration += 1
			start_node = random.randint(0, len(page_rank) - 1)
			#page_rank[start_node] += 1
	return page_rank
	

# prints the title formatted
def getTitle(title):
	return string.capwords(title.strip(string.punctuation))
	

# replaces uncommon unicode characters	
def cleanText(text):
	text = text.replace('“','"').replace('”','"').replace('–','-').replace('’','\'').replace('—','-')
	text = text.replace('’', '\'')
	return text
	

# Gets the average character length of sentences in the article
def averageSentenceLength(sentences):
	sum = 0
	for i in range(len(sentences)):
		sum += len(sentences[i])
	return int(sum / len(sentences))
	


