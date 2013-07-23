#!/usr/bin/env python
SETTINGS = {
#	'METHOD': 'classify',
	'METHOD': 'regress',
#	'EST': 'gbm',
	'EST': 'rf',
	'GRIDSEARCH': False,
	'IMPORTANCES': True,
#	'IMPORTANCES': False,
	'DIAG': False
}
params = {
	# Random Forest
	'rf': {
		'max_features': 'auto', # 3,
#			'n_estimators': 40,
# 			'min_samples_split': 1,
# 			'min_samples_leaf': 1,
	},
	# GBM
	'gbm': {
		'n_estimators': 350,
		'learning_rate': 1e-03,
		'max_depth': 3,
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
from collections import Counter, OrderedDict
from pprint import pprint
from sklearn.grid_search import GridSearchCV
if SETTINGS['METHOD'] == 'classify':
	if SETTINGS['EST'] == 'rf':
		from sklearn.ensemble import RandomForestClassifier as ESTIMATOR
	elif SETTINGS['EST'] == 'gbm':
		from sklearn.ensemble import GradientBoostingClassifier as ESTIMATOR
else:
	if SETTINGS['EST'] == 'rf':
		from sklearn.ensemble import RandomForestRegressor as ESTIMATOR
	elif SETTINGS['EST'] == 'gbm':
		from sklearn.ensemble import GradientBoostingRegressor as ESTIMATOR

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

def diag(questionset):
 	TESTCHK = Counter()
	for qn in questionset:
		qntopics = [t['name'] for t in qn['topics']]
		if qn['context_topic'] is not None and qn['context_topic']['name'] in qntopics:
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
  			TESTCHK[qn['__ans__']] += 1
#	------------
# 	lens = []
# #	limited_distribution = frozenset([''])
# 	topics = set()
# 	for qn in train:
# 		lens.append(len(qn['topics']))
# #		topics |= set([a['name'] for a in qn['topics']])
# # 		topics |= set([tuple(a.values()) for a in qn['topics']])
#   		
# # 	pprint(dict(Counter(lens)))
# # 	pprint(topics)
# #  	pprint(len(topics))
	print(sum(TESTCHK.values()), sum(TESTCHK.values()) / float(len(questionset)))
 	pprint({k: (v, v/float(sum(TESTCHK.values()))) for k, v in TESTCHK.items()})

def gen_features(questionset, givefeat=False):
	X = []
	for qn in questionset:
		f = {}
 		f['anon'] = int(qn['anonymous'])
		if 'num_answers' in qn and 'promoted_to' in qn:
			f['num_answers'] = qn['num_answers']
			f['promoted_to'] = qn['promoted_to']
		# -- our own features
		#### Topic-based ####
		if qn['context_topic'] is not None and qn['context_topic'] not in qn['topics']:
			qn['topics'].append(qn['context_topic'])
		f['topic_cnt'] = len(qn['topics'])
 		f['topic_needs'] = int(any(t['name'].startswith('Needs') for t in qn['topics']))
 		### Topic Name-based #
 		tnames = [t['name'] for t in qn['topics']]
 		f['topic_name_avglen'] = sum(len(tn) for tn in tnames)/len(tnames) if tnames else 0
 		#f['topic_name_avgtokens'] = sum(len(tokenize(tn)) for tn in tnames)/len(tnames) if tnames else 0
 		### Topic Follower-based ###
 		tfollowers = sorted([t['followers'] for t in qn['topics']])
		f['topic_ttl_followers'] = sum(tfollowers)
		#f['topic_ttl_followers_300'] = sum(tf for tf in tfollowers if tf > 300)
		#f['topic_ttl_followers_top3'] = sum(tfollowers[:3])
		#f['topic_ttl_followers_top5'] = sum(tfollowers[:5])
		#topics_with_1500_followers = sum([1 for tf in tfollowers if tf > 1500])
		#f['topics_with_1500_followers'] = topics_with_1500_followers
# 		topics_with_2000_followers = sum([1 for tf in tfollowers if tf > 2000])
# 		f['topics_with_2000_followers'] = topics_with_2000_followers
 		#topics_with_2500_followers = sum([1 for tf in tfollowers if tf > 2500])
 		#f['topics_with_2500_followers'] = topics_with_2500_followers
		### Context-topic based ###
		f['pritopic_followers'] = qn['context_topic']['followers'] if qn['context_topic'] is not None else 0

		#### Question Text based ####
		question_text_l = qn['question_text'].lower()
		f['qn_char_len'] = len(question_text_l)
		#f['qn_text_delete'] = int('delete' in question_text_l)
		### Word/Token-based ###
		tokens = tokenize(qn['question_text'])
		f['qn_token_cnt'] = len(tokens)
		f['qn_token_avglen'] = sum(len(tk) for tk in tokens) / float(len(tokens)) if len(tokens) else 0
		f['qn_token_allcaps'] = sum(1 for tk in tokens if len(tk) > 2 and tk == tk.upper())
		#f['qn_token_allcaps_per'] = sum(1 for tk in tokens if len(tk) > 1 and tk == tk.upper())/float(len(tokens))
		X.append(f.values())
	if givefeat:
		return np.array(X), f.keys()
	else:
		return np.array(X)

def get_ans(questionset):
	return np.array([int(qn['__ans__']) for qn in questionset])
	
def classify(clf, X_test):
	pred = clf.predict(X_test)
	if SETTINGS['METHOD'] == 'classify':
		return [bool(p == 1) for p in pred]
	return pred

def grid(clf, params_grid, X, Y, folds, **kwargs):
	clf_grid = GridSearchCV(clf, params_grid, cv=folds, pre_dispatch='2*n_jobs', verbose=1, refit=True, **kwargs)
	clf_grid.fit(X, Y)
	return clf_grid

def get_clf(X_train, Y_train, feat_indices=None, clf_used='rf', grid_search=False):
	params_fixed = {
		'rf': {
			'random_state': 100,
#			'verbose': 1,
			'verbose': 0,
 			'compute_importances': SETTINGS['IMPORTANCES']
		},
		'gbm': {
			'min_samples_split': 1,
			'min_samples_leaf': 2,
			'subsample': 0.5,
			'verbose': 0
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
 		if SETTINGS['IMPORTANCES'] and clf_used == 'rf':
			importances = clf.feature_importances_
			indices = np.argsort(importances)[::-1]
			print_err("Feature ranking:")
			for f, indf in enumerate(indices):
				print_err("{0}. feature {1}: {2} ({3})".format(f + 1, indf, feat_indices[indf], importances[indf]))
	 	print_err("trained!")
		return clf

def main():
	qtrain = read_set()
	X_train, featkeys = gen_features(qtrain, givefeat=True)
	Y_train = get_ans(qtrain)
	qtest = read_set()
	X_test = gen_features(qtest)

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