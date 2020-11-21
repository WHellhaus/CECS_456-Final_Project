import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

from wordcloud import WordCloud   # pip install wordcloud
from math import log, sqrt


# This function loads spam data. The format of dataset is described in
# assignment document.
# Input: file name
# Output: entire dataset, filterd by hams data, filtered by spams data
# Note all outputs are in Pandas' dataframe format
def LoadData(file_name):
	games = pd.read_csv(file_name, header=0, encoding='latin-1')
	data = [games['moves'], games['opening_eco']]
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


# This function concatenate all email bodies into a string
# Input is a data in pandas' dataframe format
# Output: string of all emails
def DataframetoString(data):
	str_data = ""
	for i, row in data.iterrows():
			str_data = str_data + str(row[2])
	return str_data

# This function loads standard NLTK stop-words and adds 
# new stop words from stopwords.dat file.
# You can modify stopword.dat any way you lile
# Input: Stop words file
# Output: A python set containing stop words
def LoadStopWords(stopwords_file):
	stop_words=set(stopwords.words("english"))
	sw_df = pd.read_csv(stopwords_file, header=None, sep=' ')

	for index, row in sw_df.iterrows():	
		stop_words.add(str(row[0]))

	return stop_words


# This function calculates the requency of words using NLTK
# Input: data in string format
# Output: data_dist is a data dictionary like NLTK object
def MostCommonWords(data):
	# tokenized_data = word_tokenize(data)
	
	# ps = PorterStemmer()

	# filtered_data = []
	# for w in tokenized_data:
	# 	if w not in stop_words:
	# 		filtered_data.append(ps.stem(w.lower()))
	
	data_dist = FreqDist(data)

	# You can use below for plotting common words
	#print("Hams:")
	#print(hams_dist.most_common(20))
	#print("Spams")
	#print(spams_dist.most_common(20))
	
	# import matplotlib.pyplot as plt
	# hams_dist.plot(30,cumulative=False)
	# spams_dist.plot(30,cumulative=False)
	# plt.show()
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
	print(p_cat_given_moves)
	return p_cat_given_moves.index(max(p_cat_given_moves))
	
def Classify(email, stop_words, p_spam, p_ham, spam_train, ham_train, numWords, alpha):
	ps = PorterStemmer()
	words = str(email).lower()
	words = word_tokenize(words)
	filtered=[]
	for w in words:
		if w not in stop_words:
			filtered.append(ps.stem(w))

	p_spam_given_email = p_spam
	p_ham_given_email = p_ham

	for word in filtered:
		p_spam_given_email *= Prob_Word_GivenY(word, spam_train, numWords, alpha, 1)
		p_ham_given_email *= Prob_Word_GivenY(word, ham_train, numWords, alpha, 0)
	print('Probabilities: ', (p_spam_given_email), p_ham_given_email)
	return 1 if (p_spam_given_email) > p_ham_given_email else 0
	
def Training2(train_data, train_wc, categories, test_data):
	dictionary = set()
	for frqdist in train_wc:
		dictionary = dictionary.union(set(frqdist.keys()))
	# print(len(dictionary))

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
	
	for i, row in test_data.iterrows():
		prediction = Classify2(row['moves'], p, train_splits, sum(num_words), alpha, categories)
		print(prediction, row['opening'])
	pass

def Training(traindata_hams, traindata_spams, hams_wc, spams_wc, test_data, stop_words):
	dictionary = list(set(hams_wc.keys()).union(set(spams_wc.keys())))
	hams_train = traindata_hams.to_numpy()
	spams_train = traindata_spams.to_numpy()

	m_hams = hams_train.shape[0]
	m_spams = spams_train.shape[0]
	m = m_hams + m_spams
	alpha = 1
	
	p_hams = (m_hams + 1) / (m + 2*alpha)
	p_spams = (m_spams + 1) / (m + 2*alpha)
	numW_hams = len(hams_wc)
	numW_spams = len(spams_wc)

	ps = PorterStemmer()

	hams_feat_list = []
	for row in hams_train:
		words = str(row[2]).lower()
		words = word_tokenize(words)
		filtered=[]
		for w in words:
			if w not in stop_words:
				filtered.append(ps.stem(w))
		row[2] = filtered
		map = {}
		for word in dictionary:
			map[word]=row[2].count(word)
		hams_feat_list.append(map)
	hams_train = np.append(hams_train, np.array(hams_feat_list).reshape(m_hams, 1), axis=1)

	spams_feat_list = []
	for row in spams_train:
		words = str(row[2]).lower()
		words = word_tokenize(words)
		filtered=[]
		for w in words:
			if w not in stop_words:
				filtered.append(ps.stem(w))
		row[2] = filtered
		map = {}
		for word in dictionary:
			map[word]=row[2].count(word)
		spams_feat_list.append(map)
	spams_train = np.append(spams_train, np.array(spams_feat_list).reshape(m_spams, 1), axis=1)
			
	test_data = test_data.to_numpy()
	correct = 0
	# print('Class: ', Classify(test_data[0][2], stop_words, p_spams, p_hams, spams_train, hams_train, (numW_hams+numW_spams), alpha))
	for row in test_data:
		prediction = Classify(row[2], stop_words, p_spams, p_hams, spams_train, hams_train, (numW_hams+numW_spams), alpha)
		print(prediction)
		correct += 1 if prediction == row[3] else 0
	print(correct / test_data.shape[0])

def main():
	games, mcw = LoadData('train.csv')
	opening_cats = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
	test, mcw_test = LoadData('test.csv')
	# print(games.head(5))
	# print(mcw)
	Training2(games, mcw, opening_cats, test)


	# mails, train_hams, train_spams = LoadData('spam_train.csv')
	# stop_words = LoadStopWords("stopwords.dat")
	# traindata_hams_str = DataframetoString(train_hams)
	# traindata_spams_str = DataframetoString(train_spams)

	# mcw_hams   = MostCommonWords(traindata_hams_str, stop_words)
	# mcw_spams  = MostCommonWords(traindata_spams_str, stop_words)

	# import matplotlib.pyplot as plt
	# mcw_hams.plot(30,cumulative=False)
	# mcw_spams.plot(30,cumulative=False)
	# plt.show()
	
	# test_data, test_hams, test_spams = LoadData("spam_test.csv")
	# Training(train_hams, train_spams, mcw_hams, mcw_spams, test_data, stop_words)	

	# wordcloud = WordCloud().generate(train_spams.iloc[:,2].str.cat())
	# plt.imshow(wordcloud, interpolation='bilinear')
	# plt.axis("off")
	# plt.show()
	
main()