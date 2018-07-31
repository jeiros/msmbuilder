.PHONY: clean

timescales:
	python timescales.py
	python timescales-plot.py

microstates:
	python microstate.py
	python microstate-plot.py


traj: msm_dict.pkl
	python microstate-traj.py

sources: msm_dict.pkl
	python sources-and-sinks.py

validate:
	python validate-msm.py
	python validate-msm-plot.py

clean:
	rm -rf *bak*
