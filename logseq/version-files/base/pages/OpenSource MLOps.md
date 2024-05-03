- Tools and best practices to help ML.
- Kuberenetes and Docker. packaging and delivering them
- ML project lifecycle: proof of concept.
- DVC: data versioning
- ![courseintro1689623628102.pdf](../assets/courseintro1689623628102_1691426697899_0.pdf)
- pip freeze > requirements.txt
- https://ritza.co/articles/dvc-s3-set-up-s3-as-dvc-remote/
-
- MLOPS Goals
	- Achieve best performance
	- Ensure reproducibility
		- transparency and team collabration
		- audits
	- Minimal setup and dependcy of 3rd party services
		- vendor lock in
		- maintence cost
		- security
- DVC helps ensure versioning of everything dataset models other artifacts
- DVC pipeline
	- data_load
	- featurize
	- data split
		- Evaluate
		- Train
- a sequence of python or Jupyter
- once all pipeline is ready exec like such
	- ``` pipline
	  dvc exp run -S train.params.n_estimators=120
	  ```
- dvc dag: dag = directed acyclic graph
- dvc exp run : run the experiment .
- dvc lock file: ensures reproducability of runs.
- ```experiment
  dvc exp run -S data_split.test_size=0.2
  ```
- ```experiment
  k
  ```
- --queue queues all experiments that will run with the same parameter
- run all option to run all experiments .
- dvc exp show
- CI/CD for ML
	- CML: continuous machine learning
		- command line tool
		- github actions:
			- yaml files defined in a special directory , tells workflows when to runs
	- Common Problems:
	- dvc repro
	-