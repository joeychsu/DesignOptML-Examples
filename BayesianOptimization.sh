

for space_dim in 5 10 15 20 ; do 
	for func_name in ackley rastrigin rosenbrock sphere ; do 
		for optimizer_type in GP RF ; do 
			python BayesianOptimization.py \
				--optimizer_type ${optimizer_type} --n_iterations 300 \
				--space_dim ${space_dim} --func_name ${func_name}
		done
	done
done

