import taso as ts
import onnx
import os
import argparse
import re

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def resnext_block(graph, input, strides, out_channels, groups):
    w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    w2 = graph.new_weight(dims=(out_channels,t.dim(1)//groups,3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    w3 = graph.new_weight(dims=(2*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*2):
        w4 = graph.new_weight(dims=(out_channels*2,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return graph.relu(graph.add(input, t))



#here we need to parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--alpha", help="alpha", default = 1.0)
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
input = graph.new_input(dims=(1,3,224,224))
weight = graph.new_weight(dims=(64,3,7,7))
t = graph.conv2d(input=input, weight=weight, strides=(2,2),
                 padding="SAME", activation="RELU")


t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="SAME")

for i in range(3):
    t = resnext_block(graph, t, (1,1), 128, 32)


strides = (2,2)
for i in range(4):
    t = resnext_block(graph, t, strides, 256, 32)
    strides = (1,1)


strides = (2,2)
for i in range(6):
    t = resnext_block(graph, t, strides, 512, 32)
    strides = (1,1)


strides = (2,2)
for i in range(3):
    t = resnext_block(graph, t, strides, 1024, 32)
    strides = (1,1)



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



# DO OPTIMIZATION AND RECORD RESULTS
# write_result('all_results = dict()\n')
# write_result('\nall_results["sysmlpartition"] = dict()\n')
# write_result('\nall_results["sysmltrick"] = dict()\n')
# write_result('\nall_results["sampletrick"] = dict()\n')
# write_result('\nall_results["sampletrick_truenewreuse"] = dict()\n')
# write_result('\nall_results["reuse"] = dict()\n')
# write_result('\nall_results["prune"] = dict()\n')

write_result('\nall_results = model_results["resnext50' + 'b' + str(budget) + '"]\n')

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

    # # for prune
    # write_result('result = dict()\n')
    # new_graph = ts.optimize_prune(graph, alpha=1.05, budget=budget, print_subst = True)
    # write_result('all_results["prune"][repeat_time] = result\n')


# STORE THE RESULTS IN THE MODEL_RESULTS VAR
# write_result('\nmodel_results["resnext50' + 'b' + str(budget) + '"] = all_results\n')


