import taso as ts
import onnx
import os
import argparse
import re

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def inceptionA(graph, input, inputC, channels):
	weight1 = graph.new_weight(dims=(64, inputC, 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(48,inputC, 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(64,48, 5, 5))
	t2 = graph.conv2d(input=t2, weight=weight3, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(64,inputC, 1, 1))
	t3 = graph.conv2d(input=input, weight=weight4, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight5 = graph.new_weight(dims=(96,64, 3, 3))
	t3 = graph.conv2d(input=t3, weight=weight5, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(96,96, 3, 3))
	t3 = graph.conv2d(input=t3, weight=weight6, strides=(1,1),
                 padding="SAME", activation="RELU")
	t4 = graph.avgpool2d(input=input, kernels=(3, 3), strides=(1,1), padding="SAME", activation = "RELU")
	weight7 = graph.new_weight(dims=(channels,inputC, 1, 1))
	t4 = graph.conv2d(input=t4, weight=weight7, strides=(1,1),
                 padding="SAME", activation="RELU")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	t12 = graph.concat(1, inputs1)
	inputs2 = list()
	inputs2.append(t3)
	inputs2.append(t4)
	t34 = graph.concat(1, inputs2)
	inputs3 = list()
	inputs3.append(t12)
	inputs3.append(t34)
	return graph.concat(1, inputs3)



def inceptionB(graph, input):
	weight1 = graph.new_weight(dims=(384, input.dim(1), 3, 3))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(2, 2),
                 padding="VALID", activation="RELU")
	weight2 = graph.new_weight(dims=(64, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(96, 64, 3, 3))
	t2 = graph.conv2d(input=t2, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(96, 96, 3, 3))
	t2 = graph.conv2d(input=t2, weight=weight4, strides=(2, 2),
                 padding="VALID", activation="RELU")
	t3 = graph.avgpool2d(input=input, kernels=(3, 3), strides=(2, 2), padding="VALID")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	t12 = graph.concat(1, inputs1)
	inputs2 = list()
	inputs2.append(t12)
	inputs2.append(t3)
	return graph.concat(1, inputs2)



def inceptionC(graph, input, channels):
	weight1 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(channels, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(channels, channels, 1, 7))
	t2 = graph.conv2d(input=t2, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(192, channels, 7, 1))
	t2 = graph.conv2d(input=t2, weight=weight4, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight5 = graph.new_weight(dims=(channels, input.dim(1), 1, 1))
	t3 = graph.conv2d(input=input, weight=weight5, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(channels, channels, 7, 1))
	t3 = graph.conv2d(input=t3, weight=weight6, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight7 = graph.new_weight(dims=(channels, channels, 1, 7))
	t3 = graph.conv2d(input=t3, weight=weight7, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight8 = graph.new_weight(dims=(channels, channels, 7, 1))
	t3 = graph.conv2d(input=t3, weight=weight8, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight9 = graph.new_weight(dims=(192, channels, 1, 7))
	t3 = graph.conv2d(input=t3, weight=weight9, strides=(1, 1),
                 padding="SAME", activation="RELU")
	t4 = graph.avgpool2d(input=input, kernels=(3, 3), strides=(1, 1), padding="SAME", activation="RELU")
	weight10 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t4 = graph.conv2d(input=t4, weight=weight10, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	t12 = graph.concat(1, inputs1)
	inputs2 = list()
	inputs2.append(t3)
	inputs2.append(t4)
	t34 = graph.concat(1, inputs2)
	inputs3 = list()
	inputs3.append(t12)
	inputs3.append(t34)
	return graph.concat(1, inputs3)




def inceptionD(graph, input):
	weight1 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(320, 192, 3, 3))
	t1 = graph.conv2d(input=t1, weight=weight2, strides=(2, 2),
                 padding="VALID", activation="RELU")
	weight3 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(192, 192, 1, 7))
	t2 = graph.conv2d(input=t2, weight=weight4, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight5 = graph.new_weight(dims=(192, 192, 7, 1))
	t2 = graph.conv2d(input=t2, weight=weight5, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(192, 192, 3, 3))
	t2 = graph.conv2d(input=t2, weight=weight6, strides=(2, 2),
                 padding="VALID", activation="RELU")
	t3 = graph.maxpool2d(input=input, kernels=(3,3), strides=(2,2), padding="VALID")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	inputs1.append(t3)
	return graph.concat(1, inputs1)




def inceptionE(graph, input):
	weight1 = graph.new_weight(dims=(320, input.dim(1), 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(384, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(384, 384, 1, 3))
	t2a = graph.conv2d(input=t2, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(384, 384, 3, 1))
	t2b = graph.conv2d(input=t2, weight=weight4, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs1 = list()
	inputs1.append(t2a)
	inputs1.append(t2b)
	t2 = graph.concat(1, inputs1)
	weight5 = graph.new_weight(dims=(448, input.dim(1), 1, 1))
	t3 = graph.conv2d(input=input, weight=weight5, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(384, 448, 3, 3))
	t3 = graph.conv2d(input=t3, weight=weight6, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight7 = graph.new_weight(dims=(384, 384, 1, 3))
	t3a = graph.conv2d(input=t3, weight=weight7, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight8 = graph.new_weight(dims=(384, 384, 3, 1))
	t3b = graph.conv2d(input=t3, weight=weight8, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs2 = list()
	inputs2.append(t3a)
	inputs2.append(t3b)
	t3 = graph.concat(1, inputs2)
	t4 = graph.maxpool2d(input=input, kernels=(3,3), strides=(1, 1), padding="SAME")
	weight9 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t4 = graph.conv2d(input=t4, weight=weight9, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs3 = list()
	inputs3.append(t1)
	inputs3.append(t2)
	inputs3.append(t3)
	inputs3.append(t4)
	return graph.concat(1, inputs3)


#here we need to parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--alpha", help="alpha", default = 1.05)
# parser.add_argument("-b", "--budget", help="budget", required=True)
# parser.add_argument("-s", "--sample_size", help="sample_size",  default = 20)
# parser.add_argument("-n", "--block_num", help="block_num")
# parser.add_argument("-c", "--cuda", help="cuda device", default = 0)

# args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.cuda))


#here we need to parse arguments
parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--alpha", help="alpha", default = 1.05)
parser.add_argument("-b", "--budget", help="budget", required=True)
# parser.add_argument("-s", "--sample_size", help="sample_size")
# parser.add_argument("-n", "--block_num", help="block_num", required = True)
parser.add_argument("-c", "--cuda", help="cuda device", default = 0)
parser.add_argument("-r", "--runtimes", help="the number of runs required", required = True)
parser.add_argument("-m", "--method", help="the method to use", required = True)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.cuda))

budget=int(args.budget)
# block_num=int(args.block_num)
runtimes=int(args.runtimes)
methods=int(args.method)

# BUILD THE ORIGINAL GRAPH
graph = ts.new_graph()
input = graph.new_input(dims=(1,3,299,299))
weight = graph.new_weight(dims=(32,3,3,3))
t = graph.conv2d(input=input, weight=weight, strides=(2,2),
                 padding="VALID", activation="RELU")

weight1 = graph.new_weight(dims=(32,32,3,3))

t = graph.conv2d(input=t, weight=weight1, strides=(1,1),
                 padding="VALID", activation="RELU")

weight2 = graph.new_weight(dims=(64,32,3,3))

t = graph.conv2d(input=t, weight=weight2, strides=(1,1),
                 padding="SAME", activation="RELU")

t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="VALID")

weight3 = graph.new_weight(dims=(80,64,1,1))

t = graph.conv2d(input=t, weight=weight3, strides=(1,1),
                 padding="VALID", activation="RELU")

weight4 = graph.new_weight(dims=(192,80,3,3))

t = graph.conv2d(input=t, weight=weight4, strides=(1,1),
                 padding="VALID", activation="RELU")

t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="VALID")


t = inceptionA(graph, t, 192, 32)
t = inceptionA(graph, t, 256, 64)
t = inceptionA(graph, t, 288, 64)
t = inceptionB(graph, t)
t = inceptionC(graph, t, 128)
t = inceptionC(graph, t, 160)
t = inceptionC(graph, t, 160)
t = inceptionC(graph, t, 192)
t = inceptionD(graph, t)
t = inceptionE(graph, t)
t = inceptionE(graph, t)

t = graph.avgpool2d(input=t, kernels=(8, 8), strides=(1,1), padding="VALID")

# new_graph = ts.optimize(graph, alpha=float(args.alpha), budget=int(args.budget), print_subst = True)

import timeit

# this helper function write "to_write" to the file
def write_result(to_write):
    f = open('results.py','a')
    f.write(to_write)
    f.close()


def get_memory():
	my_pid = os.getpid()
	print(os.system("grep VmHWM /proc/" + str(my_pid)+ "/status > memory.txt"))
	print(os.system("grep VmHWM /proc/" + str(my_pid)+ "/status"))
	print(str(my_pid))
	f2 = open("memory.txt","r")
	lines = f2.readlines()
	for line3 in lines:
		pattern = r'VmHWM:\s*([0-9]+) kB'
		matchObj = re.match(pattern, line3)
		memory = int(matchObj.group(1))
		break
	return memory



# repeat_time = 1

# DO OPTIMIZATION AND RECORD RESULTS
# write_result('all_results = dict()\n')
# write_result('\nall_results["sysmlpartition"] = dict()\n')
# write_result('\nall_results["sysmltrick"] = dict()\n')
# write_result('\nall_results["sampletrick"] = dict()\n')
# write_result('\nall_results["sampletrick_truenewreuse"] = dict()\n')
# write_result('\nall_results["reuse"] = dict()\n')
# write_result('\nall_results["prune"] = dict()\n')

write_result('\nall_results = model_results["inceptionv3' + 'b' + str(budget) + '"]\n')

for repeat_time in range(runtimes, runtimes+1):
	write_result('\nrepeat_time = ' + str(repeat_time) + '\n')

	# for sampletrick with true new reuse
	# RUN THIS ALGORITHM TO PREPARE THE OP_DICT
	# if ((methods == -1) or (methods == 1)):
	# 	# write_result('result = dict()\n')
	# 	write_result('result = all_results["sampletrick_truenewreuse"][repeat_time]\n')
	# 	new_graph = ts.optimize_sampletrick_truenewreuse(graph, 3, alpha=1.05, budget=budget, print_subst = False, sample_size = 20)

	# 	#record the peak memory
	# 	write_result('result["memory"] = ' + str(get_memory()) + '\n')

		# write_result('all_results["sampletrick_truenewreuse"][repeat_time] = result\n')		

	
	# write_result('all_results["sysmlpartition"][repeat_time] = dict()\n')
	if ((methods == -1) or (methods == 2)):
		# write_result('result = dict()\n')
		write_result('result = all_results["sysmlpartition"][repeat_time]\n')
		threshold = 30
		partitions = list()
		# 
		start_time = timeit.default_timer()
		ts.graph_partition(graph, threshold, partitions = partitions)
		end_time = timeit.default_timer()
		write_result('result["partition_time"] = ' + str(end_time-start_time) + '\n')
		# 
		new_graph = ts.optimize_partition(graph, alpha = 1.05, budget = budget, print_subst = True, eraly_stop_num = -1, partitions = partitions)
		
		#record the peak memory
		write_result('result["memory"] = ' + str(get_memory()) + '\n')

		# write_result('all_results["sysmlpartition"][repeat_time] = result\n')


	# for sysmltrick without partition
	if ((methods == -1) or (methods == 3)):
		# write_result('result = dict()\n')
		write_result('result = all_results["sysmltrick"][repeat_time]\n')
		new_graph = ts.optimize_sysmltrick(graph, alpha = 1.05, budget = budget, print_subst = False, eraly_stop_num = -1)
		
		#record the peak memory
		write_result('result["memory"] = ' + str(get_memory()) + '\n')

		# write_result('all_results["sysmltrick"][repeat_time] = result\n')

	# for sampletrick
	if ((methods == -1) or (methods == 4)):
		# write_result('result = dict()\n')
		write_result('result = all_results["sampletrick_optimized"][repeat_time]\n')
		# new_graph = ts.optimize_sampletrick(graph, alpha=1.05, budget=budget, print_subst = False, sample_size = 20)
		new_graph = ts.optimize_sampletrick_newreuse_2samplestep(graph, alpha=1.05, budget=budget, print_subst = False, sample_size = 20)

		#record the peak memory
		write_result('result["memory"] = ' + str(get_memory()) + '\n')

		# write_result('all_results["sampletrick"][repeat_time] = result\n')

	# # for reuse
	# write_result('result = dict()\n')
	# new_graph = ts.optimize_reuse(graph, alpha=1.05, budget=budget, print_subst = True)
	# write_result('all_results["reuse"][repeat_time] = result\n')

	# # for prune
	# write_result('result = dict()\n')
	# new_graph = ts.optimize_prune(graph, alpha=1.05, budget=budget, print_subst = True)
	# write_result('all_results["prune"][repeat_time] = result\n')


# STORE THE RESULTS IN THE MODEL_RESULTS VAR
# write_result('\nmodel_results["inceptionv3' + 'b' + str(budget) + '"] = all_results\n')


