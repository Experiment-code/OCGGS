import taso as ts
import onnx
import os
import argparse
import re

EMBED_SIZE = 1024


def SRUNode(graph, x, c): 
	w1 = graph.new_weight(dims=(x.dim(1), EMBED_SIZE))
	x1 = graph.matmul(x, w1)
	w2 = graph.new_weight(dims=(x.dim(1), EMBED_SIZE))
	x2 = graph.matmul(x, w2)
	f = graph.sigmoid(x2)
	w3 = graph.new_weight(dims=(x.dim(1), EMBED_SIZE))
	x3 = graph.matmul(x, w3)
	r =  graph.sigmoid(x3)
	outputs = list()
	outputs.append(graph.add(graph.mul(f, c), graph.mul(f, x1)))
	outputs.append(graph.add(graph.mul(r, outputs[0]), graph.mul(r, x)))
	return outputs



#here we need to parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--alpha", help="alpha", default = 1.05)
# parser.add_argument("-b", "--budget", help="budget", required=True)
# parser.add_argument("-s", "--sample_size", help="sample_size",  default = 20)
# parser.add_argument("-n", "--block_num", help="block_num")
# parser.add_argument("-l", "--length", help="length", default = 20)
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
parser.add_argument("-l", "--length", help="length", default = 20)
parser.add_argument("-m", "--method", help="the method to use", required = True)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.cuda))

budget=int(args.budget)
# block_num=int(args.block_num)
runtimes=int(args.runtimes)
LENGTH = int(args.length)
methods=int(args.method)
NUM_LAYERS = 1

# BUILD THE ORIGINAL GRAPH
graph = ts.new_graph()

xs = list()
for i in range(LENGTH):
	xs.append(graph.new_input(dims=(1, EMBED_SIZE)))

c = graph.new_input(dims=(1, EMBED_SIZE))

sru=list()
sru.append(list())
for i in range(LENGTH):
	if i == 0:
		sru[0].append(SRUNode(graph, xs[i], c))
	else:
		sru[0].append(SRUNode(graph, xs[i], sru[0][i-1][0]))	




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

write_result('\nall_results = model_results["sru' + str(LENGTH) + 'b' + str(budget) + '"]\n')

for repeat_time in range(runtimes, runtimes+1):
    write_result('\nrepeat_time = ' + str(repeat_time) + '\n')
    # write_result('all_results["sysmlpartition"][repeat_time] = dict()\n')
    
    # # for sampletrick with true new reuse
    # # RUN THIS ALGORITHM TO PREPARE THE OP_DICT
    # if ((methods == -1) or (methods == 1)):
    #     # write_result('result = dict()\n')
    #     write_result('result = all_results["sampletrick_truenewreuse"][repeat_time]\n')
    #     new_graph = ts.optimize_sampletrick_truenewreuse(graph, 3, alpha=1.05, budget=budget, print_subst = False, sample_size = 20)

    #     #record the peak memory
    #     write_result('result["memory"] = ' + str(get_memory()) + '\n')

    #     # write_result('all_results["sampletrick_truenewreuse"][repeat_time] = result\n')       

    
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

    # for reuse
    if ((methods == -1) or (methods == 5)):
        # write_result('result = dict()\n')
        write_result('result = all_results["reuse"][repeat_time]\n')
        new_graph = ts.optimize_reuse(graph, alpha=1.05, budget=budget, print_subst = True)

        #record the peak memory
        write_result('result["memory"] = ' + str(get_memory()) + '\n')

        # write_result('all_results["reuse"][repeat_time] = result\n')

    # for prune
    if ((methods == -1) or (methods == 6)):
        # write_result('result = dict()\n')
        write_result('result = all_results["prune"][repeat_time]\n')
        new_graph = ts.optimize_prune(graph, alpha=1.05, budget=budget, print_subst = True)

        #record the peak memory
        write_result('result["memory"] = ' + str(get_memory()) + '\n')

        # write_result('all_results["prune"][repeat_time] = result\n')



# STORE THE RESULTS IN THE MODEL_RESULTS VAR
# write_result('\nmodel_results["sru' + str(LENGTH) + 'b' + str(budget) + '"] = all_results\n')


