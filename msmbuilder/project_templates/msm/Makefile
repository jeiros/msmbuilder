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

validate:
	python validate-msm.py
	python validate-msm-plot.py

clean:
	rm -rf *bak*
