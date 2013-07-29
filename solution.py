#!/usr/bin/env python
QUESTION = 'answered'
#QUESTION = 'interest'
# QUESTION = 'views'
QN_PARAMS = {
	'answered': {
		'METHOD':				'classify',
		'n_estimators_rf':		40,
		'max_features_rf':		'auto',
		'n_estimators_gbm':		350,
		'learning_rate_gbm':	5e-03,
		'max_depth_gbm':		3,
		'features_select':		15,
		'FEATURES':				[
  									'anon',
									'topic_cnt',
									'topic_name_avglen',
# 									'topic_name_maxlen',
# 									'topic_name_avgtokens',
									'topic_ttl_followers',
# 									'topic_ttl_followers_300',
#									'topic_ttl_followers_top3',
#									'topic_ttl_followers_top5',
#									'topics_with_1500_followers',
 									'topics_with_2000_followers',
# 									'pritopic_exists',
 									'pritopic_followers',
									'pritopic_name_len',
# 									'pritopic_name_tokencnt',
# 									'pritopic',
# 									'topic_highest',
 									'topic_limited',
									'qn_char_len',
# 									'qn_token_first', # pushes it up to 0.464 but high running time
 									'qn_word_*',
#  									'qn_token_cnt',
#  									'qn_token_avglen',
#  									'qn_token_allcaps'
								]
	},
	'interest': {
		'METHOD':				'regress',
		'n_estimators_rf':		40,
		'max_features_rf':		'auto',
		'n_estimators_gbm':		250,
		'learning_rate_gbm':	5e-02,
		'max_depth_gbm':		4,
		'features_select':		15,
		'FEATURES':				[
  									'anon',
									'topic_cnt',
									'topic_name_avglen',
# 									'topic_name_maxlen',
# 									'topic_name_avgtokens',
									'topic_ttl_followers',
# 									'topic_ttl_followers_300',
#									'topic_ttl_followers_top3',
#									'topic_ttl_followers_top5',
#									'topics_with_1500_followers',
 									'topics_with_2000_followers',
# 									'pritopic_exists',
 									'pritopic_followers',
									'pritopic_name_len',
# 									'pritopic_name_tokencnt',
# 									'pritopic',
# 									'topic_highest',
# 									'topic_limited',
									'qn_char_len',
# 									'qn_token_first', # pushes it up to 0.464 but high running time
 									'qn_word_*',
#  									'qn_token_cnt',
#  									'qn_token_avglen',
#  									'qn_token_allcaps'
								]
	},
	'views': {
		'METHOD':				'regress',
		'FEATURES':				[
									'num_answers',
									'promoted_to',
  									'anon',
									'topic_cnt',
									'topic_name_avglen',
# 									'topic_name_maxlen',
# 									'topic_name_avgtokens',
									'topic_ttl_followers',
# 									'topic_ttl_followers_300',
#									'topic_ttl_followers_top3',
#									'topic_ttl_followers_top5',
#									'topics_with_1500_followers',
 									'topics_with_2000_followers',
# 									'pritopic_exists',
 									'pritopic_followers',
									'pritopic_name_len',
# 									'pritopic',
# 									'topic_highest',
 									'topic_limited',
# 									'pritopic_name_tokencnt',
##									'qn_char_len',
# 									'qn_token_first', # pushes it up to 0.464 but high running time
 									'qn_word_*',
# 									'qn_token_cnt',
# 									'qn_token_avglen',
# 									'qn_token_allcaps'
								],
		'n_estimators_rf':		60,
		'max_features_rf':		5,
		'n_estimators_gbm':		270,
# 		'n_estimators_gbm':		310,
		'learning_rate_gbm':	5e-02,
		'max_depth_gbm':		5,
		'features_select':		15,
	}
}
SETTINGS = {
	'METHOD': QN_PARAMS[QUESTION]['METHOD'],
#	'EST': 'gbm',
   	'EST': 'rf',
# 	'EST': 'ridge',
#	'EST': 'SVR',
#	'EST': 'SGD',
# 	'EST': 'lasso',
#	'EST': 'elastic',
	'EXTRA': False,
	'GRIDSEARCH': False,
	'IMPORTANCES': True,
	'DIAG': False
}
params = {
	# Random Forest
	'rf': {
		'max_features': QN_PARAMS[QUESTION]['max_features_rf'], # 3,
		'n_estimators': QN_PARAMS[QUESTION]['n_estimators_rf'],
#		'min_samples_split': 1,
#		'min_samples_leaf': 1,
	},
	# GBM
	'gbm': {
 		'n_estimators': QN_PARAMS[QUESTION]['n_estimators_gbm'],
		'learning_rate': QN_PARAMS[QUESTION]['learning_rate_gbm'],
		'max_depth': QN_PARAMS[QUESTION]['max_depth_gbm'],
		'max_features': 13
	},
	'lasso': {
 		'alpha': 2.0
#		'n_alphas': 10
	},
	'elastic': {
		'alpha': 4.0,
		'l1_ratio': 1.0
	},
	'ridge': {
		'alpha': 2.0,
	},
	'SVR': {
		'kernel': 'linear',
		'C': 2.0,
		'epsilon': 16.0,
		'tol': 0.1
	},
	'logr': {
		'tol': 1e-8,
		'penalty': 'l2',
		'C':4
	},
	'SGD': {
#		'alpha': 0.01
		'learning_rate': 'optimal'
	}
}
params_grid = {
	'rf': {
# 			'min_samples_split': [1, 2],
# 			'min_samples_leaf': [1, 2],
# 			'n_estimators': [130, 200, 250, 300, 500, 750, 1000, 1250], # [130, 400, 1000]
		'max_features': [2, 3, 4, 5] #, 6, 7, 8, 9] # [4, 6, 9]
	},
	'gbm': {
		'n_estimators': [100, 1000, 10000],
		'learning_rate': [1e-04, 1e-03, 1e-02],
		'max_depth': [2, 3, 4, 5, 6, 7]
	}
}

import sys
import json
import re
import numpy as np
from functools import partial
from collections import Counter
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.feature_selection import SelectPercentile
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
if SETTINGS['METHOD'] == 'classify':
	if SETTINGS['EST'] == 'rf':
		if SETTINGS['EXTRA']:
			from sklearn.ensemble import ExtraTreesClassifier as ESTIMATOR
		else:
			from sklearn.ensemble import RandomForestClassifier as ESTIMATOR
	elif SETTINGS['EST'] == 'gbm':
		from sklearn.ensemble import GradientBoostingClassifier as ESTIMATOR
else:
	if SETTINGS['EST'] == 'rf':
		if SETTINGS['EXTRA']:
	 		from sklearn.ensemble import ExtraTreesRegressor as ESTIMATOR
	 	else:
			from sklearn.ensemble import RandomForestRegressor as ESTIMATOR
	elif SETTINGS['EST'] == 'gbm':
		from sklearn.ensemble import GradientBoostingRegressor as ESTIMATOR
	elif SETTINGS['EST'] == 'lasso':
		from sklearn.linear_model import Lasso as ESTIMATOR
	elif SETTINGS['EST'] == 'elastic':
		from sklearn.linear_model import ElasticNet as ESTIMATOR
	elif SETTINGS['EST'] == 'ridge':
		from sklearn.linear_model import Ridge as ESTIMATOR
	elif SETTINGS['EST'] == 'SVR':
		from sklearn.svm import SVR as ESTIMATOR
	elif SETTINGS['EST'] == 'SGD':
		from sklearn.linear_model import SGDRegressor as ESTIMATOR
# 	elif SETTINGS['EST'] == 'logr':
# 		from sklearn.linear_model import LogisticRegression(

def print_err(*args):
    sys.stderr.write(' '.join(map(str,args)) + '\n')

def tokenize(txt):
	return re.sub("[^\w]", " ", txt).split()

def read_set():
	f = sys.stdin
 	stt = []
	N = int(f.readline())
	for i in xrange(N):
		stt.append(json.loads(f.readline()))
	return stt

limited_distribution = frozenset([
	"Needs Improvement",
	"Needs Attention",
	"Quora Community",
	"Describe X in N Words",
	"Joke Question",
	"Test Question",
	"Top Writers on Quora",
	"Specific Quora Users",
	"Needs More Information",
	"Needs to Be Clearer",
	"Needs Spelling, Grammar, Capitalization, or Formatting Edits",
	"Needs to Be Phrased as a Question",
	"Needs to Be a Complete Sentence",
	"Contains Multiple Questions",
	"Needs to Be Shorter",
	"Too Reliant on Question Details",
	"Rhetorical Question",
	"Needs to Be Phrased More Neutrally",
	"Advertisement, Classified Ad or Self-Promotional Question",
	"Possible Spam",
	"Needs to Be Written in English",
	"Poll Question",
	"Too Many Topics",
	"Needs Topic Adjustment",
	"Direct Question",
	"Obsolete Question",
	"Answered in Question Details",
	"Contains Unnecessary Profanity",
	"Specific Legal Question",
	"Specific Medical Question",
	"Too Broad For Reviews",
	"Possibly Insincere Question"
])

def diag(questionset):
 	TESTCHK = Counter()
 	spos, sneg = [], []
	for qn in questionset:
		qntopics = [t['name'] for t in qn['topics']]
		if any(tn in limited_distribution for tn in qntopics):
# 		if qn['context_topic'] is not None and qn['context_topic']['name'] in qntopics:
## 		if qn['question_text'].lower().startswith('is'):
#  		if len(qn['topics']) == 0:
#  		if len(qn['topics']) == 1:
# 		topics_with_n_followers = sum([1 for t in qn['topics'] if t['followers'] > 1500])
# 		if topics_with_n_followers > 2:
# 		if total_followers < 2000:
# 		if 'delete' in qn['question_text'].lower():
#  		if '?' not in qn['question_text']:
# 		if qn['context_topic'] is None:
# 		if qn['question_text'].count('?') > :
# 		if len(qn['question_text']) < 30:
# 			topics |= set([tuple(a.values()) for a in qn['topics']])
			if QN_PARAMS[QUESTION]['METHOD'] == 'classify':
				TESTCHK[qn['__ans__']] += 1
	  		else:
	  			spos.append(qn['__ans__'])
	  	elif QN_PARAMS[QUESTION]['METHOD'] == 'regress':
			sneg.append(qn['__ans__'])
#	------------
# 	lens = []
# 	topics = set()
# 	for qn in train:
# 		lens.append(len(qn['topics']))
# #		topics |= set([a['name'] for a in qn['topics']])
# # 		topics |= set([tuple(a.values()) for a in qn['topics']])
#   		
# # 	pprint(dict(Counter(lens)))
# # 	pprint(topics)
# #  	pprint(len(topics))
	if QN_PARAMS[QUESTION]['METHOD'] == 'classify':
		print(sum(TESTCHK.values()), sum(TESTCHK.values()) / float(len(questionset)))
		pprint({k: (v, v/float(sum(TESTCHK.values()))) for k, v in TESTCHK.items()})
	else:
		print_err(len(spos), sum(spos) / float(len(spos)))
		print_err(len(sneg), sum(sneg) / float(len(sneg)))
		print_err(len(questionset), (sum(spos)+sum(sneg)) / float(len(questionset)))

class CustomFeat:
	words_to_check = ['why', 'is']
	words_to_check += ['can', 'what', 'with', 'if', 'how']
	#words_to_check += ['who', 'when', 'where', 'do', 'will']
	dv = DictVectorizer(sparse=bool(SETTINGS['EST'] not in ['rf', 'gbm']))

	use_dv = True

# 	def __init__(self):
# 		for X in args:
# 			dv.fit(X)
# 		return [dv.transform(X) for X in args]

	def get_feature_names(self):
		if self.use_dv:
			return self.dv.get_feature_names()
		else:
			return QN_PARAMS[QUESTION]['FEATURES']

	def _gen_features(self, questionset):
		X = []
		if 'qn_word_*' in QN_PARAMS[QUESTION]['FEATURES']:
			QN_PARAMS[QUESTION]['FEATURES'].remove('qn_word_*')
			QN_PARAMS[QUESTION]['FEATURES'].extend(['qn_word_'+word for word in self.words_to_check])
		for qn in questionset:
			f = {}
			f['anon'] = int(qn['anonymous'])
			if 'num_answers' in qn and 'promoted_to' in qn:
				f['num_answers'] = qn['num_answers']
				f['promoted_to'] = qn['promoted_to']
			# -- own features
			#### Topic-based ####
			if qn['context_topic'] is not None and qn['context_topic'] not in qn['topics']:
				qn['topics'].append(qn['context_topic'])
			f['topic_cnt'] = len(qn['topics'])
			#f['topic_needs'] = int(any(t['name'].startswith('Needs') for t in qn['topics']))
			f['topic_limited'] = int(any(t['name'] in limited_distribution for t in qn['topics']))
			### Topic Name-based #
			tnames = [t['name'] for t in sorted(qn['topics'], key=lambda x: x['followers'], reverse=True)]
			tnamelens = [len(tn) for tn in tnames]
			f['topic_name_avglen'] = sum(tnamelens)/len(tnames) if tnames else 0
			f['topic_name_maxlen'] = max(tnamelens) if tnames else 0
			#f['topic_name_avgtokens'] = sum(len(tokenize(tn)) for tn in tnames)/len(tnames) if tnames else 0
			f['topic_highest'] = tnames[0] if tnames else 'none'
			### Topic Follower-based ###
			#tfollowers = sorted([t['followers'] for t in qn['topics']])
			tfollowers = [t['followers'] for t in qn['topics']]
			f['topic_ttl_followers'] = sum(tfollowers)
			#f['topic_ttl_followers_300'] = sum(tf for tf in tfollowers if tf > 300)
			#f['topic_ttl_followers_top3'] = sum(tfollowers[:3])
			#f['topic_ttl_followers_top5'] = sum(tfollowers[:5])
			#topics_with_1500_followers = sum([1 for tf in tfollowers if tf > 1500])
			#f['topics_with_1500_followers'] = topics_with_1500_followers
			topics_with_2000_followers = sum([1 for tf in tfollowers if tf > 2000])
			f['topics_with_2000_followers'] = topics_with_2000_followers
			#topics_with_2500_followers = sum([1 for tf in tfollowers if tf > 2500])
			#f['topics_with_2500_followers'] = topics_with_2500_followers
			### Context-topic based ###
			f['pritopic_exists'] = int(qn['context_topic'] is not None)
			if qn['context_topic'] is not None:
				f['pritopic'] = qn['context_topic']['name']
				f['pritopic_followers'] = qn['context_topic']['followers']
				f['pritopic_name_len'] = len(qn['context_topic']['name'])
				#f['pritopic_name_tokencnt'] = len(tokenize(qn['context_topic']['name']))
			else:
				f['pritopic'] = 'none'
				f['pritopic_followers'] = 0
				f['pritopic_name_len'] = 0
				#f['pritopic_name_tokencnt'] = 0
	
			#### Question Text based ####
			question_text_l = qn['question_text'].lower()
			f['qn_char_len'] = len(question_text_l)
			#f['qn_text_delete'] = int('delete' in question_text_l)
			### Word/Token-based ###
 			tokens = tokenize(qn['question_text'])
 			tokens_l = [tok.lower() for tok in tokens]
# 			f['qn_token_first'] = tokens_l[0] if len(tokens) else 'none'
 			f['qn_token_cnt'] = len(tokens)
			for word in self.words_to_check:
 				f['qn_word_'+word] = question_text_l.count(word)
#  				f['qn_word_'+word] = tokens_l.count(word)
# 				
 			f['qn_token_avglen'] = sum(len(tk) for tk in tokens) / float(len(tokens)) if len(tokens) else 0
 			f['qn_token_allcaps'] = sum(1 for tk in tokens if len(tk) > 2 and tk == tk.upper())
# 			#f['qn_token_allcaps_per'] = sum(1 for tk in tokens if len(tk) > 1 and tk == tk.upper())/float(len(tokens))
			
			#X.append([f[qt] for qt in QN_PARAMS[QUESTION]['FEATURES']])
			X.append({qt:f[qt] for qt in QN_PARAMS[QUESTION]['FEATURES']})
		return X
	
	def fit_transform(self, _X, y=None):
		X = self._gen_features(_X)
		if self.use_dv:
			return self.dv.fit_transform(X)
		else:
			return np.array(X)

	def transform(self, _X, y=None):
		X = self._gen_features(_X)
		if self.use_dv:
			return self.dv.transform(X)
		else:
			return np.array(X)

def get_ans(questionset):
	return np.array([int(qn['__ans__']) for qn in questionset])
	
def classify(clf, X_test):
	pred = clf.predict(X_test)
	if SETTINGS['METHOD'] == 'classify':
		return [bool(p == 1) for p in pred]
	else:
		return [max(p, 0.) for p in pred]

def grid(clf, params_grid, X, Y, folds, **kwargs):
	clf_grid = GridSearchCV(clf, params_grid, cv=folds, pre_dispatch='2*n_jobs', verbose=1, refit=True, **kwargs)
	clf_grid.fit(X, Y)
	return clf_grid

def get_clf(X_train, Y_train, feat_indices=None, clf_used='rf', grid_search=False):
	params_fixed = {
		'rf': {
			'random_state': 100,
			'verbose': 1,
# 			'verbose': 0,
 			'compute_importances': SETTINGS['IMPORTANCES']
		},
		'gbm': {
			'random_state': 101,
			'min_samples_split': 1,
			'min_samples_leaf': 2,
			'subsample': 0.5,
			'verbose': 0
		},
		'lasso': {
# 			'verbose': 1
		},
		'SGD': {
			'verbose': 1
		},
		'elastic': {
		},
		'SVR': {
			'verbose': True
		}
	}
	for k, v in params_fixed.iteritems():
		params[k].update(v)

	clf = ESTIMATOR()
	clf.set_params(**params[clf_used])
	if grid_search:
		return grid(clf, params_grid[clf_used], X_train, Y_train, 3)
	else:
	 	print_err("training start")
 		clf.fit(X_train, Y_train)
 		if SETTINGS['IMPORTANCES']:
 			if clf_used in ['rf', 'lasso']:
				importances = clf.feature_importances_ if clf_used == 'rf' else clf.coef_
				indices = np.argsort(importances)[::-1]
				print_err("Feature ranking:")
				for f, indf in enumerate(indices):
					print_err("{0}. feature {1}: {2} ({3})".format(f + 1, indf, feat_indices[indf].encode("utf-8"), importances[indf]))
			else:
				for i, fk in enumerate(feat_indices):
					print_err("{0}.".format(i+1), fk)

	 	print_err("trained!")
		return clf

def f_regression_(X,Y):
	import sklearn
#    a = sklearn.feature_selection.f_regression(X,Y,center=False)
#    print_err('f_reg')
#    print_err(a)
#    return a
	return sklearn.feature_selection.f_regression(X,Y,center=False)

#partial(f_regression, center=False)

def exa(b):
	return b['question_text'].lower()

def main():
	qtrain = read_set()
# 	X_train = gen_features(qtrain)
	Y_train = get_ans(qtrain)
	qtest = read_set()
# 	X_test = gen_features(qtest)
# 	(X_train, X_test), featkeys = dictVec(X_train, X_test)
	
#  	tfidf_word = TfidfVectorizer(preprocessor=lambda x: x['question_text'].lower(), ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
 	tfidf_word = TfidfVectorizer(preprocessor=exa, ngram_range=(1, 3), analyzer="word", binary=False, min_df=0.05)
#  	feat_select = SelectPercentile(score_func=f_regression_, percentile=0.15)
 	feat_select = SelectKBest(score_func=f_regression_, k=QN_PARAMS[QUESTION]['features_select'])
 	cf = CustomFeat()
 	feat = FeatureUnion([('word_counts', tfidf_word), ('custom', cf)])
# 	feat = FeatureUnion([('custom', cf)])
# 	feat = FeatureUnion([('word_counts', tfidf_word)])
 # 	est = ESTIMATOR(**params[SETTINGS['EST']])
  	w_model = Pipeline([('funion', feat), ('feat_select', feat_select)]) #, ('est', est)]
#   	w_X_train = tfidf_word.fit_transform(qtrain)
#   	w_X_test = tfidf_word.transform(qtest)
#   	print_err(w_X_train[0])
#  	X_train = w_X_train
#  	X_test = w_X_test
#  	featkeys = tfidf_word.get_feature_names()
# 	feat_select
# 	f_regression_(X_train[:,0],Y_train)
#	print_err('fitting')
#	w_model.fit(qtrain, Y_train)
# 	print_err(feat_select.get_support(indices=True))
	X_train = w_model.fit_transform(qtrain, Y_train).toarray()
	X_test = w_model.transform(qtest).toarray()
   	featkeys = np.asarray(feat.get_feature_names())[feat_select.get_support(indices=True)]
#	featkeys = []
# 	Y_test = classify(w_model, qtest)
# 	print_err(est.coef_.nonzero())

 	clf = get_clf(X_train, Y_train, feat_indices=featkeys, clf_used=SETTINGS['EST'], grid_search=SETTINGS['GRIDSEARCH'])
 	Y_test = classify(clf, X_test)
	for qn, pans in zip(qtest, Y_test):
		print json.dumps({
			'question_key': qn['question_key'].encode('ascii'),
			'__ans__': pans
		})

if SETTINGS['DIAG']:
	qtrain = read_set()
	diag(qtrain)
elif __name__ == "__main__":
	main()