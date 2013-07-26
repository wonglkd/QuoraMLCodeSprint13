answered.pred: solution.py answered/input00.in
	time ./solution.py < answered/input00.in > $@

interest.pred views.pred:%.pred: solution.py %/input00.in
	time ./solution.py < $*/input00.in > $@

d-%: solution.py %/input00.in
	time ./$< < $*/input00.in

p-%: solution.py %/input00.in
	python -m cProfile -s 'time' $< < $*/input00.in

c-answered: check.py answered/output00.out answered.pred
	./$^

c-interest c-views:c-%: check.py %/output00.out %.pred
	./$^ --rmsle