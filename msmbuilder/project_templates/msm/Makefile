.PHONY: clean

timescales:
	python timescales.py
	python timescales-plot.py

microstates:
	python microstate.py
	python microstate-plot.py
	python microstate-traj.py

sources: msm_dict.pkl
	python sources_and_sinks.py

clean:
	rm -rf *bak*
	rm *pdf
