.PHONY : clean
all:
	python tica.py
plot: ttrajs
	python tica-plot.py
clean:
	rm tica.pkl
	rm -rf ttrajs*
	rm *bak*
sample: tica-sample-coordinate.py
	python tica-sample-coordinate.py
	python tica-sample-coordinate-plot.py
