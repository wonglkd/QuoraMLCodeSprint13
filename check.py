#!/usr/bin/env python
import argparse
import json
import numpy as np
import math

def rmsle(preds, actuals):
	preds, actuals = np.array(preds), np.array(actuals)
	return 0.5 / math.sqrt(np.sum((np.log(preds + 1.) - np.log(actuals + 1.)) ** 2) / len(actuals))

def readf(filename):
	with open(filename, 'rb') as f:
		for line in f:
			qn = json.loads(line)
			yield qn

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('goldfile')
	parser.add_argument('testfile')
	parser.add_argument('-r', '--rmsle', action='store_true')
	args = parser.parse_args()
	ans = {}
	correct, wrong = 0, 0

	for qn in readf(args.goldfile):
		ans[qn['question_key']] = qn['__ans__']

	if args.rmsle:
		preds, actuals = [], []
		for qn in readf(args.testfile):
			preds.append(qn['__ans__'])
			actuals.append(ans[qn['question_key']])
		print 'Score:', rmsle(preds, actuals)
	else:
		for qn in readf(args.testfile):
			if ans[qn['question_key']] == qn['__ans__']:
				correct += 1
			else:
				wrong += 1
		print 'Correct:', correct
		print 'Wrong:', wrong
		print 'Score:', correct/float(correct+wrong)

if __name__ == "__main__":
	main()