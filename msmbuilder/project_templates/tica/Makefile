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

sparse:
	python sparse-tica.py

sparseplot: sptica_list.pkl
	python sparse-tica-plot.py

