import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 


def LoadData():
	to_exclude = [i for i in range(2, 19800)]# specifies the amount of data to load in (leave list empty to load all data)
	games = pd.read_csv('games.csv', header=0, encoding='latin-1', skiprows=to_exclude)
	opening_cats = {'A00-A39': 0, 'A40-A44': 1, 'A45-A49': 2, 'A50-A79': 3, 'A80-A99': 4, 
		'B00-B19': 5, 'B20-B99': 6, 'C00-C19': 7, 'C20-C99': 8, 'D00-D69': 9, 'D70-D99': 10, 'E00-E59': 11, 'E60-E99': 12}
	labels = []
	for index, row in games.iterrows():
		opening_num = int(row['opening_eco'][1:])
		if (row['opening_eco'][0] == 'A'):
			if(opening_num <= 39):
				labels.append(0)
			elif(opening_num <= 44):
				labels.append(1)
			elif(opening_num <= 49):
				labels.append(2)
			elif(opening_num <= 79):
				labels.append(3)
			else:
				labels.append(4)
		elif (row['opening_eco'][0] == 'B'):
			if(opening_num <= 19):
				labels.append(5)
			else:
				labels.append(6)
		elif (row['opening_eco'][0] == 'C'):
			if(opening_num <= 19):
				labels.append(7)
			else:
				labels.append(8)
		elif (row['opening_eco'][0] == 'D'):
			if(opening_num <= 69):
				labels.append(9)
			else:
				labels.append(10)
		else:
			if(opening_num <= 59):
				labels.append(11)
			else:
				labels.append(12)
		# labels.append(opening_cats[row['opening_eco'][0]])
	games = pd.concat([games, pd.DataFrame({'label': labels})], axis=1)
	headers = list(games.columns.values)

	X_train, X_test, y_train, y_test = train_test_split(games.to_numpy(), labels, test_size=0.2)
	X_train = pd.DataFrame(data=X_train, columns=headers)
	X_test = pd.DataFrame(data=X_test, columns=headers)

	# dictionary for how to tokenize moves into a list
	# by ply: split by  each move of white or black
	# by turn: split by each turn i.e. one white move and one black move
	# by turn with number: split by turn and add the number of the turn to the beginning of the string (psuedo-dependency)
	move_tokenizer_options = {'by ply': 0, 'by turn': 1, 'by turn with number': 2}

	games, mcw = processGames(X_train, True, move_tokenizer_options['by turn'], opening_cats)
	test, mcw_test = processGames(X_test, False, move_tokenizer_options['by turn'], opening_cats)
	# print(test)
	return games, test, mcw

def processGames(games, truncate_ply, move_tokenizer, opening_cats):
	data = [games['moves'], games['opening_eco']]
	ply = games['opening_ply']
	labels = games['label']
	headers = ['moves', 'opening']
	data = pd.concat(data, axis=1, keys=headers)
	maxPly = 14

	for index, row in data.iterrows():
		row['opening'] = labels[index]
		ply[index] = ply[index]+1 if (ply[index] % 2 != 0) else ply[index]
		moveCount = 0
		moves = word_tokenize(row['moves'])

		if (move_tokenizer == 0):
			if(truncate_ply):
				row['moves'] = moves[0:ply[index]+1]
			else:
				row['moves'] = moves[0:maxPly+1]
		else:
			formattedMoves = []
			for move in moves:
				if(truncate_ply): 
					if (moveCount >= (ply[index])):
						break
				elif (moveCount >= maxPly):
					break

				if(move_tokenizer == 1):
					if(moveCount%2==0):
						formattedMoves.append(str(move))
					else:
						formattedMoves[int(moveCount/2)] += ' ' + str(move)
				if(move_tokenizer == 2):
					if(moveCount%2==0):
						formattedMoves.append(str(int(moveCount/2)+1) + '.' + str(move))
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
	return p_cat_given_moves.index(max(p_cat_given_moves))
	
def Training2(train_data, train_wc, categories, test_data):
	dictionary = set()
	for frqdist in train_wc:
		dictionary = dictionary.union(set(frqdist.keys()))
	# print(len(dictionary))

	# m = [len(x[1]) for x in train_data.groupby('opening')]

	alpha = 1

	num_words = [len(frqdist) for frqdist in train_wc]

	train_data['feature_list'] = ""
	for i, row in train_data.iterrows():
		word_map = {}
		for word in dictionary:
			word_map[word] = row['moves'].count(word)
		row['feature_list'] = word_map

	# train_splits = [x[1] for x in train_data.groupby('opening')]
	train_splits = []
	m = []
	for key in categories:
		rows = train_data.loc[train_data['opening'] == categories[key]]
		m.append(len(rows))
		train_splits.append(rows)
	p = [(m_cat + 1) / (sum(m) + len(categories)*alpha) for m_cat in m]
	
	correct = 0
	shape = np.zeros(shape=(len(categories), len(categories)))
	conf_matrix = pd.DataFrame(shape)
	for i, row in test_data.iterrows():
		prediction = Classify2(row['moves'], p, train_splits, sum(num_words), alpha, categories)
		conf_matrix.iat[prediction, row['opening']] += 1
		correct += 1 if prediction == row['opening'] else 0
		# print(prediction, row['opening'])
	print('ACCURACY: ', correct/len(test_data))
	print(conf_matrix)
	

def main():
	opening_cats = {'A00-A39': 0, 'A40-A44': 1, 'A45-A49': 2, 'A50-A79': 3, 'A80-A99': 4, 
		'B00-B19': 5, 'B20-B99': 6, 'C00-C19': 7, 'C20-C99': 8, 'D00-D69': 9, 'D70-D99': 10, 'E00-E59': 11, 'E60-E99': 12}
	games, test,  mcw = LoadData()
	Training2(games, mcw, opening_cats, test)
	
main()