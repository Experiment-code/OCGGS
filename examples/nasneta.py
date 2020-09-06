import taso as ts
import onnx
import os
import argparse
import re

def squeeze(graph, out_channels, input):
    weight = graph.new_weight(dims=(out_channels, input.dim(1), 1, 1))
    return graph.conv2d(input=input, weight=weight,
                        strides=(1, 1), padding="SAME",
                        activation="RELU")


def fit(graph, current, input):
    if input.dim(2) == current.dim(2):
        return squeeze(graph, current.dim(1), input)
    else:
        weight = graph.new_weight(dims=(current.dim(1), input.dim(1), 3, 3))
        return graph.conv2d(input=input, weight=weight, strides=(2, 2), padding="SAME", activation="RELU")


def seperable_conv(graph, input, out_channels, kernels, strides, padding, activation = "NONE"):
    assert input.dim(1) % out_channels == 0, "input.dim(1)={}, out_channels={}".format(input.dim(1), out_channels)
    weight1 = graph.new_weight(dims=(out_channels, input.dim(1) // out_channels, kernels[0], kernels[1]))
    t = graph.conv2d(input=input, weight=weight1, strides=strides, padding=padding)
    weight2 = graph.new_weight(dims=(out_channels, t.dim(1), 1, 1))
    return graph.conv2d(input=t, weight=weight2, strides=(1, 1), padding="SAME", activation=activation)


def normal_cell(graph, prev, cur, out_channels):
    cur = squeeze(graph, out_channels, cur)
    prev = fit(graph, cur, prev)
    ts = list()
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(cur)
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(graph.avgpool2d(input=cur, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(prev)
    ts.append(graph.avgpool2d(input=prev, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(graph.avgpool2d(input=prev, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    assert len(ts) == 10, "Expected 10 tensors, got {}".format(len(ts))
    outputs = list()
    for i in range(5):
        outputs.append(graph.add(ts[2*i], ts[2*i+1]))
    return graph.concat(1, outputs)


def reduction_cell(graph, prev, cur, out_channels):
    cur = squeeze(graph, out_channels, cur)
    prev = fit(graph, cur, prev)
    ts = list()
    outputs = list()
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(7,7), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(5,5), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[0], ts[1]))
    ts.append(graph.maxpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(7,7), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[2], ts[3]))
    ts.append(graph.avgpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(5,5), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[4], ts[5]))
    ts.append(graph.maxpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=outputs[0], out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    outputs.append(graph.add(ts[6], ts[7]))
    ts.append(graph.avgpool2d(input=outputs[0], kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(outputs[1])
    outputs.append(graph.add(ts[8], ts[9]))
    return graph.concat(1, outputs)



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
input = graph.conv2d(input=input, weight=weight, strides=(2,2),
                 padding="SAME", activation="RELU")


input = graph.maxpool2d(input=input, kernels=(3,3), strides=(2,2), padding="SAME")

out_channels = 128
for i in range(3):
    prev = input
    cur = input
    for j in range(5):
        t = normal_cell(graph, prev, cur, out_channels)
        prev = cur
        cur = t
    out_channels *= 2
    input = reduction_cell(graph, prev, cur, out_channels)



# new_graph = ts.optimize(graph, alpha=1.0, budget=-1)

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

write_result('\nall_results = model_results["nasneta' + 'b' + str(budget) + '"]\n')

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

    # # for reuse
    # write_result('result = dict()\n')
    # new_graph = ts.optimize_reuse(graph, alpha=1.05, budget=budget, print_subst = True)
    # write_result('all_results["reuse"][repeat_time] = result\n')

    # # for prune
    # write_result('result = dict()\n')
    # new_graph = ts.optimize_prune(graph, alpha=1.05, budget=budget, print_subst = True)
    # write_result('all_results["prune"][repeat_time] = result\n')


# STORE THE RESULTS IN THE MODEL_RESULTS VAR
# write_result('\nmodel_results["nasneta' + 'b' + str(budget) + '"] = all_results\n')


