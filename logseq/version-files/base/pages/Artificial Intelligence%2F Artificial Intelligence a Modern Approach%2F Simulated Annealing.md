- np hard: non deterministic polynomial time.
- Iterative improvement problems: TSP travelling salesman problem
- N queens:
	- N!/(N-2)!2!
- Hill Climbing:
	- Local maximum/minimum
	- Taboo search to combat that
	- stage algorithm
	- both are random restarts
	- step size tuning:
		- start with large step size and reduce over time: Simulated annealing
- Simulated Annealing:
	- ```python
	  while True:
	    current_state
	    T = temperature_measure(t)
	    if T =0:
	      return current_state
	    next = random successor
	    if  delta > 0:
	      current = next
	    else:
	      current = next with probability e**delta/T
	    
	  
	  ```
- local beam search:
	- track and keep best particles
	- ![image.png](../assets/image_1724968495028_0.png)
	- stochastic beam search:
		- best are not based only on fitness but with some randomness
- Genetic Algorithms:
	- ![image.png](../assets/image_1724969206990_0.png)
	- GA Crossover
		-