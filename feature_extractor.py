# extract features from list of text instances based on configuration set of features

from nltk.corpus import stopwords
import nltk
import re
import os
import numpy as np
import operator
import string

def function_words(tokenized_list):
	'''
	:param tokenized_list of words from N texts
	:return: NxM numpy array with M counts for the english stop words for the N texts
	Header values of each stop word
	'''
	bow = []
	header = stopwords.words('english')
	for tokens in tokenized_list:
		counts = []
		for sw in header:
			sw_count = tokens.count(sw)
			normed = sw_count/float(len(tokens))
			counts.append(normed)
		bow.append(counts)
	bow_np = np.array(bow).astype(float)
	return bow_np, header	

def lexical(tokenized_list):
	lwc = []
	header = get_top_30(tokenized_list)
	for tokens in tokenized_list:
		counts = []
		for lw in header:
			lw_count = tokens.count(lw)
			normed = lw_count/float(len(tokens))
			counts.append(normed)
		lwc.append(counts)
	lwc_np = np.array(lwc).astype(float)
	return lwc_np, header

def syntax(tokenized_list):
	pos_counts = []
	header = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS','CC','PRP','VB','VBG']
	for tokens in tokenized_list:
		counts = []
		pos_tags = nltk.pos_tag(tokens)
		pos_tags = [tag[1] for tag in pos_tags]
		for tag in header:
			syn_count = pos_tags.count(tag)
			normed = syn_count/float(len(tokens))
			counts.append(normed)
		pos_counts.append(counts)
	pos_np = np.array(pos_counts).astype(float)
	return pos_np, header

def punctuation(tokenized_list):
	punct_counts = []
	punct = string.punctuation
	headers = list(punct)
	for tokens in tokenized_list:
		counts = []
		pos_tags = nltk.pos_tag(tokens)
		pos_tags = [tag[1] for tag in pos_tags]
		for char in punct:
			punct_count = tokens.count(char)
			normed = punct_count / float(len(tokens))
			counts.append(normed)
		punct_counts.append(counts)
	pos_np = np.array(punct_counts).astype(float)
	print(headers)
	return pos_np, headers

def complexity(tokenized_list,texts):
	complexity_count = []
	#headers = ['avg_chars', 'uniq_words', 'avg_word', 'long_words']
	headers = ['long_words']
	for i,tokens in enumerate(tokenized_list):
		counts = []
		long_words_count = 0
		avg_chars = 0
		word_count = len(tokens)
		for token in tokens:
			avg_chars += len(token)
			if len(token) > 6:
				long_words_count+=1
		avg_chars = float(avg_chars)/word_count
		sents = nltk.sent_tokenize(texts[i])
		avg_word = float(len(tokens))/len(sents)
		#counts.append(avg_chars)
		#counts.append(float(len(set(tokens)))/word_count)
		#counts.append(avg_word)
		counts.append(long_words_count)
		complexity_count.append(counts)
	pos_np = np.array(complexity_count).astype(float)
	return pos_np, headers

def get_top_30(tokenized_list):
	fname = 'top30.txt'
	if os.path.isfile(fname):
		with open(fname) as f:
			content = f.readlines()
		top30_words = [x.strip() for x in content]
	else:
		all_tokens = [token.lower() for tokens in tokenized_list for token in tokens if token.lower() not in stopwords.words('english') and token[0] not in string.punctuation]
		freq_dist = nltk.FreqDist(all_tokens)
		print(freq_dist)
		sorted_fdist = sorted(freq_dist.items(), key=operator.itemgetter(1), reverse=True)
		print(sorted_fdist)
		top30_words = []
		for key, val in sorted_fdist[:30]:
			pos_tag = nltk.tag.pos_tag([key])
			top30_words.append(key+"\t"+pos_tag)
		with open(fname,'w') as f:
			for item in top30_words:
				f.write("%s\n" % item)
	return top30_words

def tokenize_texts(texts):
	tokenized_list = []
	for text in texts:
		tokenized_list.append(nltk.word_tokenize(text))
	return tokenized_list

def extract_features(texts, conf):
	features = []
	headers = []
	print("Starting to extract features")
	tokenized_list = tokenize_texts(texts)
	print("Completed tokenization")
	top30_words = get_top_30(tokenized_list)
	print("Got top 30 words")

	if 'function_words' in conf:
		f,h = function_words(tokenized_list)
		features.append(f)
		headers.extend(h)

	if 'syntax' in conf:
		f,h = syntax(tokenized_list)
		features.append(f)
		headers.extend(h)

	if 'lexical' in conf:
		f,h = lexical(tokenized_list)
		features.append(f)
		headers.extend(h)

	if 'complexity' in conf:
		f, h = complexity(tokenized_list,texts	)
		features.append(f)
		headers.extend(h)

	if 'punctuation' in conf:
		f,h = punctuation(tokenized_list)
		features.append(f)
		headers.extend(h)

	all_features = np.concatenate(features,axis=1)
	return all_features, headers
