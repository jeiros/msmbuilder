.PHONY : clean
cluster:
	python cluster.py
	python cluster-plot.py
sample: top.pdb
	python sample-clusters.py
	python sample-clusters-plot.py
clean:
	rm *pkl.bak*
	rm -rf *.bak*/
	rm -rf cluster_samples
	rm cluster-samples.pdf
	rm load-cluster-samples.tcl
