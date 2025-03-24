


for space_dim in 5 10 15 20 ; do 
	for func_name in ackley rastrigin rosenbrock sphere ; do 
		for model_type in mlp rf svr gp ; do 
			python SurrogateBasedOptimization.py \
				--n_iterations 300 --space_dim ${space_dim} \
				--func_name ${func_name} --model_type ${model_type}
		done
	done
done
