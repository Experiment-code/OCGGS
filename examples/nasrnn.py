import taso as ts
import onnx
import os
import argparse
import re

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

hidden_size = 512
length = 5

def combine(graph, x, h):
    w1 = graph.new_weight(dims=(hidden_size, x.dim(1)))
    w2 = graph.new_weight(dims=(hidden_size, h.dim(1)))
    return graph.add(graph.matmul(x, w1), graph.matmul(h, w2))


def nas_node(graph, input, x):
    t = list()
    for i in range(8):
        t.append(combine(graph, x, input))
    midt = list()
    midt.append(graph.add(graph.relu(t[0]), graph.sigmoid(t[3])))
    midt.append(graph.add(graph.sigmoid(t[1]), graph.tanh(t[2])))
    midt.append(graph.mul(graph.sigmoid(t[4]), graph.tanh(t[5])))
    midt.append(graph.mul(graph.sigmoid(t[6]), graph.relu(t[7])))
    midt.append(graph.add(graph.sigmoid(midt[1]), graph.tanh(midt[2])))
    midt.append(graph.mul(graph.tanh(midt[0]), graph.tanh(midt[3])))
    midt.append(graph.mul(graph.tanh(midt[4]), graph.tanh(midt[5])))
    return graph.tanh(midt[6])



#here we need to parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--alpha", help="alpha", default = 1.05)
# parser.add_argument("-b", "--budget", help="budget", required=True)
# parser.add_argument("-s", "--sample_size", help="sample_size",  default = 20)
# parser.add_argument("-n", "--block_num", help="block_num")
# parser.add_argument("-l", "--length", help="length", default = 5)
# parser.add_argument("-c", "--cuda", help="cuda device", default = 0)

# args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.cuda))

# length = int(args.length)


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
xs = list()
for i in range(length):
    xs.append(graph.new_input(dims=(1, hidden_size)))


state = graph.new_weight(dims=(1, hidden_size))

for i in range(length):
    state = nas_node(graph, state, xs[i])


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

write_result('\nall_results = model_results["nasrnn' + 'b' + str(budget) + '"]\n')


for repeat_time in range(runtimes, runtimes+1):
    write_result('\nrepeat_time = ' + str(repeat_time) + '\n')

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
        new_graph = ts.optimize_partition(graph, alpha = 1.05, budget = budget, print_subst = True, eraly_stop_num = -1, partitions = partitions, do_weight_process=False)
        
        #record the peak memory
        write_result('result["memory"] = ' + str(get_memory()) + '\n')

        # write_result('all_results["sysmlpartition"][repeat_time] = result\n')


    # for sysmltrick without partition
    if ((methods == -1) or (methods == 3)):
        # write_result('result = dict()\n')
        write_result('result = all_results["sysmltrick"][repeat_time]\n')
        new_graph = ts.optimize_sysmltrick(graph, alpha = 1.05, budget = budget, print_subst = False, eraly_stop_num = -1, do_weight_process=False)
        
        #record the peak memory
        write_result('result["memory"] = ' + str(get_memory()) + '\n')

        # write_result('all_results["sysmltrick"][repeat_time] = result\n')

    # for sampletrick
    if ((methods == -1) or (methods == 4)):
        # write_result('result = dict()\n')
        write_result('result = all_results["sampletrick_optimized"][repeat_time]\n')
        # new_graph = ts.optimize_sampletrick(graph, alpha=1.05, budget=budget, print_subst = False, sample_size = 20)
        new_graph = ts.optimize_sampletrick_newreuse_2samplestep(graph, alpha=1.05, budget=budget, print_subst = False, sample_size = 20, do_weight_process=False)

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
# write_result('\nmodel_results["nasrnn' + 'b' + str(budget) + '"] = all_results\n')


