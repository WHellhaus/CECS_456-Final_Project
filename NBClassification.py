import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

from wordcloud import WordCloud   # pip install wordcloud
from math import log, sqrt


# This function loads spam data. The format of dataset is described in
# assignment document.
# Input: file name
# Output: entire dataset, filterd by hams data, filtered by spams data
# Note all outputs are in Pandas' dataframe format
# NOT USED IN THIS CODE (see processGames function instead)
def LoadData(file_name):
	games = pd.read_csv(file_name, header=0, encoding='latin-1')
	data = [games['moves'], games['opening_eco']]
	ply = games['opening_ply']
	headers = ['moves', 'opening']
	data = pd.concat(data, axis=1, keys=headers)
	# data = data.head(500)

	opening_cats = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

	for index, row in data.iterrows():
		row['opening'] = opening_cats[row['opening'][0]]
		moveCount = 0
		moves = word_tokenize(row['moves'])
		formattedMoves = []
		for move in moves:
			if(moveCount > (ply[index]*2)):
				break
			if(moveCount%2==0):
				# formattedMoves.append(str(int(moveCount/2)+1) + '.' + str(move))
				formattedMoves.append(str(move))
			else:
				formattedMoves[int(moveCount/2)] += ' ' + str(move)
			moveCount += 1
		row['moves'] = formattedMoves
	
	mcw = []
	for key in opening_cats:
		rows = data.loc[data['opening'] == opening_cats[key]]
		# print(key, rows)
		moves = []
		for index, row in rows.iterrows():
			moves += row['moves']
		mcw.append(MostCommonWords(moves))
	return data, mcw

def processGames(games, truncate_ply, opening_cats):
	data = [games['moves'], games['opening_eco']]
	ply = games['opening_ply']
	headers = ['moves', 'opening']
	data = pd.concat(data, axis=1, keys=headers)
	maxPly = 14

	for index, row in data.iterrows():
		row['opening'] = opening_cats[row['opening'][0]]
		ply[index] = ply[index]+1 if (ply[index] % 2 != 0) else ply[index]
		moveCount = 0
		moves = word_tokenize(row['moves'])
		formattedMoves = []
		for move in moves:
			if(truncate_ply): 
				if (moveCount >= (ply[index])):
					break
			elif (moveCount >= maxPly):
				break
			if(moveCount%2==0):
				# uncomment this top line to use moveCount in the moves list i.e. 1. d4 e5, 2. 
				# formattedMoves.append(str(int(moveCount/2)+1) + '.' + str(move))
				formattedMoves.append(str(move))
			else:
				formattedMoves[int(moveCount/2)] += ' ' + str(move)
			moveCount += 1
		row['moves'] = formattedMoves
	
	mcw = []
	for key in opening_cats:
		rows = data.loc[data['opening'] == opening_cats[key]]
		# print(key, rows)
		moves = []
		for index, row in rows.iterrows():
			moves += row['moves']
		mcw.append(MostCommonWords(moves))
	return data, mcw


# This function calculates the requency of words using NLTK
# Input: data in string format
# Output: data_dist is a data dictionary like NLTK object
def MostCommonWords(data):
	data_dist = FreqDist(data)
	return data_dist

def Prob_Word_GivenY(word, train_data, numWords, alpha, y):
	sum = 0
	count_y = 0
	for i, row in train_data.iterrows():
		if(row['feature_list'].get(word)):
			if(row['opening']==y and row['feature_list'].get(word)>0):
				sum += 1
				count_y += 1
	return (sum + alpha) / (count_y + numWords*alpha)

def Classify2(moves, p_category, train_splits, numWords, alpha, categories):
	p_cat_given_moves = [x for x in p_category]
	
	for move in moves:
		for key, value in categories.items():
			p_cat_given_moves[value] *= Prob_Word_GivenY(move, train_splits[value], numWords, alpha, value)
	# print(p_cat_given_moves)
	return p_cat_given_moves.index(max(p_cat_given_moves))
	
def Training2(train_data, train_wc, categories, test_data):
	dictionary = set()
	for frqdist in train_wc:
		dictionary = dictionary.union(set(frqdist.keys()))
	print(len(dictionary))

	m = [len(x[1]) for x in train_data.groupby('opening')]

	alpha = 1

	p = [(m_cat + 1) / (sum(m) + len(categories)*alpha) for m_cat in m]

	num_words = [len(frqdist) for frqdist in train_wc]

	train_data['feature_list'] = ""
	for i, row in train_data.iterrows():
		word_map = {}
		for word in dictionary:
			word_map[word] = row['moves'].count(word)
		row['feature_list'] = word_map

	train_splits = [x[1] for x in train_data.groupby('opening')]
	
	correct = 0
	for i, row in test_data.iterrows():
		prediction = Classify2(row['moves'], p, train_splits, sum(num_words), alpha, categories)
		correct += 1 if prediction == row['opening'] else 0
		print(prediction, row['opening'])
	print('ACCURACY: ', correct/len(test_data))
	

def main():
	to_exclude = [i for i in range(2, 19300)]# specifies the amount of data to load in (leave list empty to load all data)
	games = pd.read_csv('games.csv', header=0, encoding='latin-1', skiprows=to_exclude)
	opening_cats = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
	labels = []
	for index, row in games.iterrows():
		labels.append(opening_cats[row['opening_eco'][0]])
	games = pd.concat([games, pd.DataFrame({'label': labels})], axis=1)
	headers = list(games.columns.values)

	X_train, X_test, y_train, y_test = train_test_split(games.to_numpy(), labels, test_size=0.33)
	X_train = pd.DataFrame(data=X_train, columns=headers)
	X_test = pd.DataFrame(data=X_test, columns=headers)

	games, mcw = processGames(X_train, True, opening_cats)
	test, mcw_test = processGames(X_test, False, opening_cats)
	print(games.head(5))
	print(test.head(5))
	Training2(games, mcw, opening_cats, test)
	
main()