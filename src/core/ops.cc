/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * This file is modified by the author based on the original file
 */

#include "taso/ops.h"
#include "taso/substitution.h"
#include <math.h> //add for the function: exp()
#include <random> //add for sampling
#include <iostream>
#include <fstream>
using namespace std;
using namespace taso;

const Op Op::INVALID_OP = Op();
const SplitInfo SplitInfo::NO_SPLIT = SplitInfo();

/*
bool Op::operator==(const Op& b)
{
  if (guid != b.guid) return false;
  return (ptr == b.ptr);
}

bool Op::operator<(const Op& b)
{
  if (guid != b.guid) return guid < b.guid;
  return ptr < b.ptr;
}
*/

//Op::Op(void)
//{
//  guid = GUID_INVALID;
//  ptr = NULL;
//  //index = -1;
//  //par_i = -1;
//}


// the op structure for all search algorithms
Op::Op(void)
{
	guid = GUID_INVALID;
	ptr = NULL;
	SIStepOrder = -1;
	SIidx = -1;
	guid_for_cost = -1;
}

Edge::Edge(void)
: srcOp(Op::INVALID_OP), dstOp(Op::INVALID_OP), srcIdx(-1), dstIdx(-1)
{}

Edge::Edge(Op _srcOp, Op _dstOp, int _srcIdx, int _dstIdx)
: srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx)
{}

SrcEdge::SrcEdge(int _idx, Op _op)
: idx(_idx), op(_op)
{}

/*
bool Tensor::operator==(const Tensor& b)
{
  if (numDim != b.numDim) return false;
  for (int i = 0; i < numDim; i++)
    if (dim[i] != b.dim[i]) return false;
  if (idx != b.idx) return false;
  if (op.guid != b.op.guid) return false;
  return true;
}
*/

OpBase::OpBase(Model* _model, OpType _type)
: numInputs(0), model(_model), type(_type), runtime(0.0f)
{
  // Assume only constant operator can take no inputs
  assert(type == OP_CONSTANT_POOL);
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input,
               Model* _model, OpType _type)
: numInputs(1), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input0,
               const Tensor& _input1,
               Model* _model, OpType _type)
: numInputs(2), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input0,
               const Tensor& _input1,
               const Tensor& _input2,
               Model* _model, OpType _type)
: numInputs(3), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  inputs[2] = _input2;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(const Tensor& _input0,
               const Tensor& _input1,
               const Tensor& _input2,
               const Tensor& _input3,
               const Tensor& _input4,
               Model* _model, OpType _type)
: numInputs(5), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  inputs[2] = _input2;
  inputs[3] = _input3;
  inputs[4] = _input4;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(int n, Tensor* _inputs, Model* _model, OpType _type)
: numInputs(n), model(_model), type(_type), runtime(0.0f)
{
  assert(n <= MAX_NUM_INPUTS);
  for (int i = 0; i < n; i++)
    inputs[i] = _inputs[i];
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

bool OpBase::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_OP_TYPE:
      *value = (int) type;
      return true;
    case PM_NUM_INPUTS:
      *value = numInputs;
      return true;
    case PM_NUM_OUTPUTS:
      *value = numOutputs;
      return true;
    default:
      return false;
  }
}

bool OpBase::get_input_parameter(TNParameter tnp, DIMParameter dim, int* value)
{
  int inputIdx = 0, dimIdx = 0;
  switch (tnp) {
    case IN_5:
      inputIdx++;
    case IN_4:
      inputIdx++;
    case IN_3:
      inputIdx++;
    case IN_2:
      inputIdx++;
    case IN_1:
      inputIdx++;
    case IN_0:
      break;
    default:
      return false;
  }
  if (inputIdx >= numInputs) return false;
  switch (dim) {
    case DIM_3:
      dimIdx ++;
    case DIM_2:
      dimIdx ++;
    case DIM_1:
      dimIdx ++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = inputs[inputIdx].numDim;
      return true;
    default:
      return false;
  }
  if (dimIdx >= inputs[inputIdx].numDim) return false;
  *value = inputs[inputIdx].dim[dimIdx];
  return true;
}

std::string Op::op_to_string(const OpBase* ptr)
{
  switch (ptr->type) {
    case OP_INPUT:
      return "Input";
    case OP_WEIGHT:
      return "Weight";
    case OP_ANY:
      return "Any";
    case OP_CONV2D:
      return "Conv";
    case OP_DROPOUT:
      return "Dropout";
    case OP_LINEAR:
      return "Linear";
    case OP_POOL2D_MAX:
      return "MaxPool";
    case OP_POOL2D_AVG:
      return "AveragePool";
    case OP_RELU:
      return "Relu";
    case OP_SIGMOID:
      return "Sigmoid";
    case OP_TANH:
      return "TanH";
    case OP_BATCHNORM:
      return "Batchnorm";
    case OP_CONCAT:
      return "Concat";
    case OP_SPLIT:
      return "Split";
    case OP_RESHAPE:
      return "Reshape";
    case OP_TRANSPOSE:
      return "Transpose";
    case OP_EW_ADD:
      return "Add";
    case OP_EW_MUL:
      return "Mul";
    case OP_MATMUL:
      return "Matmul";
    case OP_MUL:
      return "Mul";
    case OP_ENLARGE:
      return "Enlarge";
    case OP_SQUEEZE:
      return "Squeeze";
    case OP_UNSQUEEZE:
      return "Unsqueeze";
    case OP_EW_SUB:
      return "Sub";
    case OP_EW_DIV:
      return "Div";
    case OP_EW_EQUAL:
      return "Equal";
    case OP_EW_GREATER:
      return "Greater";
    case OP_EW_LESS:
      return "Less";
    case OP_EW_MAX:
      return "Max";
    case OP_EW_MIN:
      return "Min";
    case OP_REDUCE_ARGMAX:
      return "ArgMax";
    case OP_REDUCE_ARGMIN:
      return "ArgMin";
    case OP_REDUCE_MAX:
      return "ReduceMax";
    case OP_REDUCE_MEAN:
      return "ReduceMean";
    case OP_REDUCE_MIN:
      return "ReduceMin";
    case OP_REDUCE_PROD:
      return "ReduceProd";
    case OP_REDUCE_SUM:
      return "ReduceSum";
    case OP_PAD:
      return "Pad";
    case OP_SHAPE:
      return "Shape";
    case OP_SIZE:
      return "Size";
    case OP_TOPK:
      return "TopK";
    case OP_WHERE:
      return "Where";
    case OP_CEIL:
      return "Ceil";
    case OP_CAST:
      return "Cast";
    case OP_EXP:
      return "Exp";
    case OP_ROUND:
      return "Round";
    case OP_LOG:
      return "Log";
    case OP_LOGICAL_NOT:
      return "Not";
    case OP_SQRT:
      return "Sqrt";
    case OP_LEAKYRELU:
      return "LeakyRelu";
    case OP_SLICE:
      return "Slice";
    case OP_RESIZE:
      return "Resize";
    default:
      return "Unknown_" + std::to_string(ptr->type);
  }
}

static Model* model_singleton = NULL;

Graph::Graph()
: totalCost(-1.0f)
{
  if (model_singleton == NULL) {
    model_singleton = new Model();
  }
  model = model_singleton;
  model->print_cost = false;
  //size_t inputSize = sizeof(DATATYPE) * n * c * h * w;
  //checkCUDA(cudaMalloc(&input.ptr, inputSize));
  //printf("Initialize a graph\n");
}

void Graph::print_measurements(void)
{
  model->print_cost = true;
}

TensorHandle Graph::new_input(int ndim, const int* dims)
{
  TensorHandle t = new Tensor(ndim, dims, GUID_INPUT);
  t = input_wrapper(t);
  return t;
}

TensorHandle Graph::new_weight(int ndim, const int* dims, const DATATYPE* weight_initial)
{
  DATATYPE* weight_ptr = NULL;
  int total_size = sizeof(DATATYPE);
  for (int i = 0; i < ndim; i++)
    total_size *= dims[i];
  weight_ptr = (DATATYPE*) model->allocate_memory(total_size, weight_initial);
  TensorHandle t = new Tensor(ndim, dims, GUID_WEIGHT, weight_ptr);
  t = weight_wrapper(t);
  return t;
}

TensorHandle Graph::new_weight(const Tensor& weight)
{
  TensorHandle t = new Tensor(weight);
  t->op.guid = GUID_WEIGHT;
  t->op.ptr = NULL;
  t->idx = 0;
  t->data_ptr = (DATATYPE*) model->allocate_memory(
      weight.volume() * sizeof(DATATYPE), (DATATYPE*) weight.data_ptr);
  t = weight_wrapper(t);
  return t;
}



//---------------------THIS FUNCTION OF PRINTING GRAPH SUBST HISTORY FOR ALL SEARCH ALGORITHM----------------------//
//this function prints the subst history of a graph
//graph is the Graph whose subst history needs to be print
void print_subst_history_n_cost(Graph* graph) {
	printf("\n        ===== Applied Substitutions    TOTAL COST: %.4lf =====\n", graph->total_cost());
	for (size_t i = 0; i < graph->subst_history.size(); i++) {
		printf("        substitution[%03zu] ||   type: %d ||   cost change: %.4lf\n", i, graph->subst_history[i].substType, graph->subst_history[i].cost_change);
		Graph::GraphSubst subst = graph->subst_history[i];
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
			printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
			printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
	}
}


//---------------------THIS FUNCTION FOR ALL SEARCH ALGORITHM----------------------//
//old_cost: the cost of the original computation graph
//new_cost: the cost of the optimized computation graph
//search_time: the running time of the whole search process
//matching_time: the time for finding substitution
//gentime: the time for generating new graphs and do other preparations (not include the time for sampling and finding affected ranges
//
void record_result(float old_cost, float new_cost, double search_time, double matching_time, double gen_time) {
	//first: open the file
	ofstream timer_fs;
	timer_fs.open("results.py", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//record old cost
	timer_fs << "result[\"old_cost\"] = " << old_cost << std::endl;

	//record new cost
	timer_fs << "result[\"new_cost\"] = " << new_cost << std::endl;

	//record search time
	timer_fs << "result[\"search_time\"] = " << search_time << std::endl;

	//record matching time
	timer_fs << "result[\"matching_time\"] = " << matching_time << std::endl;

	//record gentime
	timer_fs << "result[\"gen_time\"] = " << gen_time << std::endl;

	//close the file
	timer_fs.close();
}

//---------------------THIS FUNCTION OF DELETING GRAPH FOR ALL SEARCH ALGORITHM----------------------//
//ori_graph: the original graph
//to_delete: the graph to delete
//if "to_delete" is not the "ori_graph", we can delete it; otherwise, we cannot
void delete_graph(Graph* ori_graph, Graph* to_delete) {
	if (ori_graph != to_delete)
		delete to_delete;
}


//---------------------THIS STRUCTURE ONLY FOR SAMPLETRICK----------------------//
//The constructor of my helper struct
SearchTreeNodesample::SearchTreeNodesample(void)
{
	graphPtr = NULL;
	//fatherPos = -1; //an invalid position
	sample_quota = 0;
}
SearchTreeNodesample::SearchTreeNodesample(Graph* _graphPtr)
{
	graphPtr = _graphPtr;
	//fatherPos = _fatherPos;
	sample_quota = 0;
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK----------------------//
void add_to_uniq_topk(std::priority_queue<SearchTreeNodesample*, std::vector<SearchTreeNodesample*>, SearchTreeNodesampleCompare>& topknodes, size_t k_size, SearchTreeNodesample* to_add)
{
	if (topknodes.size() < k_size) {
		topknodes.push(to_add);
		//topknodes.insert(to_add);
	}
	else {
		//if ((*topknodes.begin())->graphPtr->total_cost() > to_add->graphPtr->total_cost()) {
		//	//topknodes.pop();
		//	topknodes.erase(topknodes.begin());
		//	//topknodes.push(to_add);
		//	topknodes.insert(to_add);
		//}
		if (topknodes.top()->graphPtr->total_cost() > to_add->graphPtr->total_cost()) {
			topknodes.pop();
			topknodes.push(to_add);
		}
	}
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
void add_to_uniq_topk(std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare>& topknodes, size_t k_size, SearchTreeNode* to_add)
{
	if (topknodes.size() < k_size) {
		topknodes.push(to_add);
	}
	else {
		if (topknodes.top()->graphPtr->total_cost() > to_add->graphPtr->total_cost()) {
			topknodes.pop();
			topknodes.push(to_add);
		}
	}
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
void add_to_uniq_topk(std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodepotentialCompare>& topknodes, size_t k_size, SearchTreeNode* to_add)
{
	if (topknodes.size() < k_size) {
		topknodes.push(to_add);
	}
	else {
		if (topknodes.top()->potential > to_add->potential) {
			topknodes.pop();
			topknodes.push(to_add);
		}
	}
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK----------------------//
//This function assigns the sample quota to every candidate in the new level
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes
void do_sampling(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNodesample>& candidates) {
	//std::vector<float> tem_sample_weights;
	//printf("----------start sample---------------\n");

	///////////////Use the weight of each node as the probability of it to be sampled
	//below is to find the biggest cost reduction
	//float biggest_cost_reduction = 0;
	//int num_hopeless = 0;
	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap;



	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNodesample*, std::vector<SearchTreeNodesample*>, SearchTreeNodesampleCompare> topknodes_good;
	std::priority_queue<SearchTreeNodesample*, std::vector<SearchTreeNodesample*>, SearchTreeNodesampleCompare> topknodes_bad;

	//look ahead for one step
	if (level_end == 0) {
		//this is the first level, no need to check last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			//if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			//else {
			if (tem_graph->subst_history.back().cost_change > 0)
				add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//}
		}
	}
	else {
		//need to check the last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				continue;
			}


			if (tem_graph->subst_history.back().cost_change > 0) {
				//int last_depend_step = tem_graph->subst_history.back().biggestNode.SIStepOrder;
				int last_step = tem_graph->subst_history.size() - 2;
				/*if (last_depend_step == -1)
				candidates.at(weight_i).sample_quota = 1;*/
				if (tem_graph->subst_history.at(last_step).cost_change < 0) {
					//if the cost does not increase and increase for twice
					add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
					//candidates.at(weight_i).sample_quota = 1;
				}
				else
					num_bad_bad++;
			}
			//else if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//candidates.at(weight_i).sample_quota = 1;
		}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK----------------------//
Graph* Graph::optimize_sampletrick(float alpha, int budget, bool print_subst, int sample_size)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/       
	
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);



	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//calculate the distance from the first node in a source graph to any other node in the source graph  
	//assume all source graphs are connected
	std::vector<size_t> ranges; //store the max range of each xfer
	std::vector<bool> is_connected_source_graph; //record whether the source graph is connected or not
	size_t max_range = 0;
	for (size_t i = 0; i < xfers.size(); i++) {
		size_t nCheck_op = 1; //the number of OpXs in the next range to check
		size_t start_op = 0; //the start position of OpXs in the next range
		size_t nCheck_tx = 0; //the number of TexsorXs in the next range to check
		size_t start_tx = 0; //the start position of TensorXs in the next range

		GraphXfer* xfer_to_check = xfers[i];
		std::vector<OpX*> to_Check_op; //nodes which we need to visit their neighbors
		std::unordered_set<OpX*> already_checked_op; //nodes which we  already visited
		std::vector<int> to_Check_tx; //the idxs of tensors (inputs) which we need to visit their neighbors
		std::unordered_set<int> already_checked_tx; //tensors (inputs) which we  already visited

		to_Check_op.push_back(xfer_to_check->srcOps[0]);
		already_checked_op.insert(xfer_to_check->srcOps[0]);

		size_t largest_dst_op = 0; //the largest distance between srcOps[0] and any other OpX in the source graph
		size_t largest_dst = 0; //the largest distance between srcOps[0] and any other node (including inputs) in the source graph

								//run BFS
		while (true) {
			size_t new_nCheck_op = 0;
			size_t new_nCheck_tx = 0;
			//check OpXs
			for (size_t j = 0; j < nCheck_op; j++) {
				OpX* opx_check = to_Check_op[start_op + j];
				//check all inputs of this OpX
				for (size_t in_i = 0; in_i < opx_check->inputs.size(); in_i++) {
					TensorX& temTX = opx_check->inputs[in_i]; //tem var TensorX
					if (temTX.op != NULL) {
						//not an input tensor
						if (already_checked_op.find(temTX.op) == already_checked_op.end()) {
							//if the OpX corresponding to the input has not been checked
							already_checked_op.insert(temTX.op);
							to_Check_op.push_back(temTX.op);
							new_nCheck_op++;
						}
					}
					else {
						// an input tensor
						if (already_checked_tx.find(temTX.idx) == already_checked_tx.end()) {
							already_checked_tx.insert(temTX.idx);
							to_Check_tx.push_back(temTX.idx);
							new_nCheck_tx++;
						}
					}
				}
				//check all outputs of this OpX
				for (size_t out_i = 0; out_i < opx_check->outputs.size(); out_i++) {
					TensorX& temTX = opx_check->outputs[out_i]; //tem var TensorX
					if (temTX.op != NULL) {
						//not a input tensor
						if (already_checked_op.find(temTX.op) == already_checked_op.end()) {
							//if the OpX corresponding to the input has not been checked
							already_checked_op.insert(temTX.op);
							to_Check_op.push_back(temTX.op);
							new_nCheck_op++;
						}
					}
					else {
						//an input tensor
						if (already_checked_tx.find(temTX.idx) == already_checked_tx.end()) {
							already_checked_tx.insert(temTX.idx);
							to_Check_tx.push_back(temTX.idx);
							new_nCheck_tx++;
						}
					}
				}
			}

			//check TensorXs
			for (size_t j = 0; j < nCheck_tx; j++)
			{
				int idx_check = to_Check_tx[start_tx + j];
				for (size_t opx_i = 0; opx_i < (xfer_to_check->srcOps.size()); opx_i++)
				{
					OpX* opx_check = xfer_to_check->srcOps[opx_i];
					for (size_t in_i = 0; in_i < (opx_check->inputs.size()); in_i++) {
						TensorX& temTX = opx_check->inputs[in_i];
						if (temTX.op == NULL && temTX.idx == idx_check) {
							if (already_checked_op.find(opx_check) == already_checked_op.end()) {
								already_checked_op.insert(opx_check);
								to_Check_op.push_back(opx_check);
								new_nCheck_op++;
							}
							break;
						}
					}
				}
			}

			if ((new_nCheck_op + new_nCheck_tx) > 0) {
				largest_dst++;
				if (new_nCheck_op > 0)
					largest_dst_op = largest_dst;
				start_op += nCheck_op;
				start_tx += nCheck_tx;
				nCheck_op = new_nCheck_op;
				nCheck_tx = new_nCheck_tx;
			}
			else
				break;
		}

		ranges.push_back(largest_dst_op);
		//update the is_connected_source_graph
		if (already_checked_op.size() < xfer_to_check->srcOps.size())
			is_connected_source_graph.push_back(false);
		else
			is_connected_source_graph.push_back(true);
		if (largest_dst_op > max_range)
			max_range = largest_dst_op;
	}



	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	std::deque<SearchTreeNodesample> candidates;

	std::vector<std::set<size_t>> hashmaps; //assign a hashmap for each level

											//////////////////////////
	SearchTreeNodesample rootNode(this);
	//candidates.push_back(this);
	//store the sample quota
	rootNode.sample_quota = (size_t)sample_size;
	candidates.push_back(rootNode);
	//////////////////////////

	//hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;


	Graph *bestGraph = this;
	//float bestCost = total_cost();
	//below variable records the bestCost which is updated every time a node is finding its child nodes
	float bestCost_for_sample = total_cost();
	float best_step_cost_reduc = 0; // the largest cost reduction by one step currently

									//printf("MetaFlow Cost = %.4lfms\n", bestCost);
									//printf("Input graph: end-to-end execution time =\n"
									//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();


	size_t add_counter = 1; //how many searchTreeNodes in candidates have been pushed
	size_t del_counter = 0; //how many searchTreeNodes in candidates have been deleted
	size_t reuse_counter = 0;

	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
	

	//assign a hashmap set for level 1
	std::set<size_t> hashmap_level_1;
	hashmaps.push_back(hashmap_level_1);

	//store the explored sequences as a tree in a deque
	for (size_t i = 0; i < xfers.size(); i++) {
		std::set<Op, OpCompare> empty_range;
		candidates[0].childrenNode.push_back(std::vector<size_t>());

		//RECORD THE START OF SUBST
		auto time_s = std::chrono::system_clock::now();

		xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false, gentime);
		
		//RECORD THE END OF SUBST
		auto time_e = std::chrono::system_clock::now();
		auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
		double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		run_function_time += searchtime_in;
	}


	///////////////////////////////CODE for re-allocating sample quota
	size_t level_start = 0; //the original index of the start node in a level
	size_t level_end = 0; //the original index of the end node in a level
						  ///////////////////////////////END CODE for re-allocating sample quota

						  ///////////////////////CODE record the cost range of the sampled nodes in each level
	std::vector<float> upperbound;
	std::vector<float> lowerbound;
	///////////////////////END CODE

	while (true)
	{
		SearchTreeNodesample& fatherNode = candidates.front();
		//size_t child_counter = 0; //record thich child node is being processed and finding its child nodes.

		//////////////////CODE for sampling which re-allocate the sample quota
		if (del_counter == level_start) {

			//printf("\n=============NEW LEVEL %zu=============\n", candidates.back().graphPtr->subst_history.size());

			//if father node is the start node for its level in the search tree
			//re-allocate sample quota now
			do_sampling(level_end, del_counter, sample_size, candidates);

			//update level_start and level_end
			level_start = level_end + 1;
			level_end = candidates.size() - 1 + del_counter;

			//assign a hashmap set for the next level to be searched
			std::set<size_t> hashmap_level_next;
			hashmaps.push_back(hashmap_level_next);

		}
		//////////////////END CODE

		//find possible optimization steps based on the child nodes
		for (size_t i = 0; i < fatherNode.childrenNode.size(); i++)
		{
			for (size_t j = 0; j < fatherNode.childrenNode[i].size(); j++)
			{
				//check whether this child node is sampled or not//////////////////////
				//size_t curr_sample_size = fatherNode.child_sample_quota.at(child_counter);

				//child_counter++;

				size_t tem_pos = fatherNode.childrenNode[i][j] - del_counter;
				Graph* subGraph = candidates.at(tem_pos).graphPtr;


				if (print_subst) {
					//print whether this graph is sampled or not
					printf("\n---------------SELF: %zu    FATHER: %zu    SAMPLE QUOTA: %zu-----------", fatherNode.childrenNode[i][j], del_counter, candidates.at(tem_pos).sample_quota);
					//print subst history infor for each graph, no matter whether it is sampled or nor
					print_subst_history_n_cost(subGraph);

				}
				
				//update bestCost_for_sample
				if (subGraph->total_cost() < bestCost_for_sample) {
					//delete bestGraph;
					delete_graph(original_graph, bestGraph);

					bestCost_for_sample = subGraph->total_cost();
					bestGraph = subGraph;

				}

				if (candidates.at(tem_pos).sample_quota == 0) {
					//there is no quota for this child node, i.e., it is not sampled

					if (subGraph != bestGraph) {
						//delete subGraph;
						delete_graph(original_graph, subGraph);
					}

					continue;
				}
				//////////////////////////////////////////////////////////////////////

				//candidates.pop_front();

				//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

				//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
				if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
				{
					if (subGraph != bestGraph) {
						//delete subGraph;
						delete_graph(original_graph, subGraph);
					}

					continue;
				}

				//find affected range, store as vector<set<nodes>>
				std::vector<std::vector<Op>> single_affected_ranges(max_range + 1); //every vector member stores ops between two ranges (circles)
				std::vector<std::set<Op, OpCompare>> affected_ranges(max_range + 1); //every vector member stores ops in a certain range (circle)
				for (size_t range = 0; range <= max_range; range++) {
					if (range == 0) {
						const Graph::GraphSubst& last_step = subGraph->subst_history.back();
						//GraphSubst has the attribute: std::vector<Op> srcOps, dstOps;
						for (size_t dstop_i = 0; dstop_i < last_step.dstOps.size(); dstop_i++) {
							single_affected_ranges[range].push_back(last_step.dstOps[dstop_i]);
							affected_ranges[range].insert(last_step.dstOps[dstop_i]);
						}
					}
					else {
						const std::vector<Op>& last_range = single_affected_ranges[range - 1];
						affected_ranges[range] = affected_ranges[range - 1];
						for (size_t lastop_i = 0; lastop_i < last_range.size(); lastop_i++) {
							//check in_edges
							const std::set<Edge, EdgeCompare>& lastop_in = subGraph->inEdges.at(last_range[lastop_i]); //calling at() should not throw an execption, because it will always be found
							std::set<Edge, EdgeCompare>::const_iterator in_it;
							for (in_it = lastop_in.begin(); in_it != lastop_in.end(); in_it++) {
								//only normal operator or wrappered inputs/weights can be added into the set
								if ((in_it->srcOp.ptr != NULL) && (affected_ranges[range].find(in_it->srcOp) == affected_ranges[range].end())) {
									affected_ranges[range].insert(in_it->srcOp);
									single_affected_ranges[range].push_back(in_it->srcOp);
								}
							}
							//check out_edges
							std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator lastop_out_it = subGraph->outEdges.find(last_range[lastop_i]);
							if (lastop_out_it != subGraph->outEdges.end()) {
								//if this op has outEdges
								const std::set<Edge, EdgeCompare>& lastop_out = lastop_out_it->second;
								std::set<Edge, EdgeCompare>::const_iterator out_it;
								for (out_it = lastop_out.begin(); out_it != lastop_out.end(); out_it++) {
									if (affected_ranges[range].find(out_it->dstOp) == affected_ranges[range].end()) {
										//do not need check op.ptr != NULL here, because dstOp can not be null
										affected_ranges[range].insert(out_it->dstOp);
										single_affected_ranges[range].push_back(out_it->dstOp);
									}
								}
							}
						}
					}
				}

				SearchTreeNodesample& curr_node = candidates.at(tem_pos);
				for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
					curr_node.childrenNode.push_back(std::vector<size_t>());

					//RECORD THE START OF SUBST
					auto time_s = std::chrono::system_clock::now();

					xfers[xfer_i]->run_sampletrick(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[ranges[xfer_i]], tem_pos, add_counter, del_counter, reuse_counter, best_step_cost_reduc, is_connected_source_graph[xfer_i], gentime);
					
					//RECORD THE END OF SUBST
					auto time_e = std::chrono::system_clock::now();
					auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
					double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
					run_function_time += searchtime_in;
				}

				//delete the subGraph if it is not the best
				if (subGraph != bestGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}
			}
		}

		candidates.pop_front();
		del_counter++;
		//while (!candidates.empty())
		//{
		// if (candidates.front().childrenNode.size() == 0) {
		//  candidates.pop_front();
		//  del_counter++;
		// }
		// else
		//  break;
		//}
		if (candidates.empty())
			break;
	}


	//if (print_subst) {
	// printf("        ===== Applied Substitutions =====\n\n");
	// for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
	//  printf("        substitution[%03zu]: \n", i);
	//  Graph::GraphSubst subst = bestGraph->subst_history[i];
	//  for (size_t j = 0; j < subst.srcOps.size(); j++) {
	//	  printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
	//  }
	//  for (size_t j = 0; j < subst.dstOps.size(); j++) {
	//	  printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
	//  }
	// }
	//}

	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	////record old cost
	//timer_fs << "result[old_cost] = " << original_graph->total_cost() << std::endl;

	////record new cost
	//timer_fs << "result[new_cost] = " << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime,  run_function_time - gentime, gentime);

	//std::cout << "THE TYPE OF TIME:" << typeid(double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den).name() << endl;

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	//if (print_subst) {
	if (true) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
			printf("        substitution[%03zu]: cost change: %.4lf\n", i, bestGraph->subst_history[i].cost_change);
			Graph::GraphSubst subst = bestGraph->subst_history[i];
			for (size_t j = 0; j < subst.srcOps.size(); j++) {
				printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
			}
			for (size_t j = 0; j < subst.dstOps.size(); j++) {
				printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
			}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        add_counter: %zu\n", add_counter);

	printf("        reuse_counter: %zu\n", reuse_counter);

	//////////////////CODE print cost range of sampled nodes in each level
	/*for (int cost_range_i = 0; cost_range_i < bestGraph->subst_history.size(); cost_range_i++) {
	printf("the cost range in level %d     good nodes [%.4lf, %.4lf]     bad nodes [%.4lf, %.4lf]\n", cost_range_i, lowerbound[cost_range_i * 2], upperbound[cost_range_i * 2], lowerbound[cost_range_i * 2 + 1], upperbound[cost_range_i * 2 + 1]);
	}*/
	/////////////////END CODE

	////////////////
	return bestGraph;
}




//---------------------THIS FUNCTION ONLY FOR PRUNE----------------------//
Graph* Graph::optimize_prune(float alpha, int budget, bool print_subst)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));

	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);


	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<Graph*> candidates;
	std::forward_list<Graph*> candidates;

	std::set<size_t> hashmap;

	//candidates.push_back(this);
	candidates.push_front(this); //insert this at the head of the list

	hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;

	Graph *bestGraph = this;
	float bestCost = total_cost();
	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();

	//std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::iterator it;
	//for (it = inEdges.begin(); it != inEdges.end(); it++) {
	// Op& curOp = it->first;
	// curOp.SIStepOrder = -1;
	// curOp.first.SIidx = it->first.guid;
	//}

	int counter = 0;
	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
	while (!candidates.empty()) {
		//Graph *subGraph = candidates.top();
		//candidates.pop();
		Graph *subGraph = candidates.front();
		candidates.pop_front();
		if (subGraph->total_cost() < bestCost) {
			//delete bestGraph;
			delete_graph(original_graph, bestGraph);
			
			bestCost = subGraph->total_cost();
			bestGraph = subGraph;
		}

		//////////////////////////////
		//when budget <= 0, we do not have the budget constraint, and the stopping condition is that all candidates are checked
		//if ((budget > 0)  && (counter > budget)) {
		//  // TODO: free all remaining candidates when budget exhausted 
		//  break;
		//}
		//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"

		//every graph in candidates will be counted
		//if (counter % 1 == 0) {
		//	printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
		//	//timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
		//}
		counter++;

		if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
		{
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			continue;
		}

		//std::forward_list<int>::const_iterator next_pos = candidates.before_begin();

		for (size_t i = 0; i < xfers.size(); i++) {
			//for (size_t j = 0; j < xfers[i]->srcOps.size(); j++) {
			//  printf("srcOps[%zu]: type(%d)\n", j, xfers[i]->srcOps[j]->type);
			//}
			//for (size_t j = 0; j < xfers[i]->dstOps.size(); j++) {
			//  printf("dstOps[%zu]: type(%d)\n", j, xfers[i]->dstOps[j]->type);
			//}
			
			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();
			
			xfers[i]->run_prune(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, i, gentime);

			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;
		}
		if (bestGraph != subGraph) {
			//delete subGraph;
			delete_graph(original_graph, subGraph);
		}
	}
	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	if (print_subst) {
		printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
			printf("        substitution[%03zu]: \n", i);
			Graph::GraphSubst subst = bestGraph->subst_history[i];
			for (size_t j = 0; j < subst.srcOps.size(); j++) {
				printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
			}
			for (size_t j = 0; j < subst.dstOps.size(); j++) {
				printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
			}
		}
	}

	////////////////
	//print counter
	printf("        counter: %d\n", counter);
	////////////////
	return bestGraph;
}

//---------------------THIS FUNCTION ONLY FOR ENUMERATION----------------------//
Graph* Graph::optimize_enumeration(float alpha, int budget, bool print_subst)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));

	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);


	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<Graph*> candidates;
	std::forward_list<Graph*> candidates;

	std::set<size_t> hashmap;

	//candidates.push_back(this);
	candidates.push_front(this); //insert this at the head of the list

	hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;

	Graph *bestGraph = this;
	float bestCost = total_cost();
	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();

	//std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::iterator it;
	//for (it = inEdges.begin(); it != inEdges.end(); it++) {
	// Op& curOp = it->first;
	// curOp.SIStepOrder = -1;
	// curOp.first.SIidx = it->first.guid;
	//}

	int counter = 0;
	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

						//ofstream timer_fs;
						//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

						//											 //record old cost
						//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
	while (!candidates.empty()) {
		//Graph *subGraph = candidates.top();
		//candidates.pop();
		Graph *subGraph = candidates.front();
		candidates.pop_front();
		if (subGraph->total_cost() < bestCost) {
			//delete bestGraph;
			delete_graph(original_graph, bestGraph);

			bestCost = subGraph->total_cost();
			bestGraph = subGraph;
		}

		//////////////////////////////
		//when budget <= 0, we do not have the budget constraint, and the stopping condition is that all candidates are checked
		//if ((budget > 0)  && (counter > budget)) {
		//  // TODO: free all remaining candidates when budget exhausted 
		//  break;
		//}
		//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"

		//every graph in candidates will be counted
		//if (counter % 1 == 0) {
		//	printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
		//	//timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
		//}
		counter++;

		if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
		{
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			continue;
		}

		//std::forward_list<int>::const_iterator next_pos = candidates.before_begin();

		for (size_t i = 0; i < xfers.size(); i++) {
			//for (size_t j = 0; j < xfers[i]->srcOps.size(); j++) {
			//  printf("srcOps[%zu]: type(%d)\n", j, xfers[i]->srcOps[j]->type);
			//}
			//for (size_t j = 0; j < xfers[i]->dstOps.size(); j++) {
			//  printf("dstOps[%zu]: type(%d)\n", j, xfers[i]->dstOps[j]->type);
			//}

			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();

			xfers[i]->run_enumeration(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, i, gentime);

			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;
		}
		if (bestGraph != subGraph) {
			//delete subGraph;
			delete_graph(original_graph, subGraph);
		}
	}
	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	if (print_subst) {
		printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
			printf("        substitution[%03zu]: \n", i);
			Graph::GraphSubst subst = bestGraph->subst_history[i];
			for (size_t j = 0; j < subst.srcOps.size(); j++) {
				printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
			}
			for (size_t j = 0; j < subst.dstOps.size(); j++) {
				printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
			}
		}
	}

	////////////////
	//print counter
	printf("        counter: %d\n", counter);
	////////////////
	return bestGraph;
}


//---------------------THIS STRUCTURE ONLY FOR REUSE----------------------//
/////////////////////////////////
//The constructor of my helper struct
SearchTreeNode::SearchTreeNode(void)
{
	graphPtr = NULL;
	//fatherPos = -1; //an invalid position
	father = -1;
	//grandpa = -1;
	searched = false;
	first_child = 0;
	last_child = 0;
	
	sample_quota = 0;
	further_search = false;
	has_sampled_child = false;
	reused = false;

	potential = -1;
}
SearchTreeNode::SearchTreeNode(Graph* _graphPtr, int fatherid)
{
	graphPtr = _graphPtr;
	//fatherPos = _fatherPos;
	father = fatherid;
	//grandpa = -1;
	searched = false;
	first_child = 0;
	last_child = 0;

	sample_quota = 0;
	further_search = false;
	has_sampled_child = false;
	reused = false;

	potential = -1;
}
/////////////////////////////////


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//this function updates the last_to_search pos and delete nodes in candidates if necessary
int update_last_to_search(std::deque<SearchTreeNode>& candidates, int last_to_search, Graph *bestGraph, Graph* original_graph) {
	if (!candidates.back().searched)
		return candidates.size() - 1;
	else
	{
		// in this case, the last searched node has no child nodes, since there is no new nodes added
		while (true) //until we find a node to search, or there is no node to search
		{
			if (last_to_search - 1 >= 0) {
				if (candidates.at(last_to_search - 1).searched == false) //if its father node has other child node to search 
					return last_to_search - 1;
				else
				{
					// all child nodes of its father node has been searched
					const SearchTreeNode& last_searched_node = candidates.at(last_to_search);
					size_t father_pos = last_searched_node.father;

					// we need to delete useless graph here
					for (size_t child = last_to_search; child < candidates.size(); child++) {
						if (candidates.at(child).graphPtr != bestGraph) {
							//delete candidates.at(child).graphPtr;
							delete_graph(original_graph, candidates.at(child).graphPtr);
						}
					}

					candidates.erase(candidates.begin() + last_to_search, candidates.end()); // delete all child nodes of its father_pos
					last_to_search = father_pos;
				}
			}
			else { // the last search node is the first one, then it must has been searched
				if (candidates.begin()->graphPtr != bestGraph) {
					//delete candidates.begin()->graphPtr;
					delete_graph(original_graph, candidates.begin()->graphPtr);
				}
				candidates.erase(candidates.begin());
				return -1;
			}
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//find the pos of this opx in the srcops list
size_t op2ind(OpX* to_find, const std::vector<OpX*>& srcOps) {
	for (size_t i = 0; i < srcOps.size(); i++) {
		if (to_find == srcOps[i])
			return i;
	}
	assert(false); // the function must return some value in the loop
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//this function calculate the shortest distance between any two nodes in the source graph of the input substitution
//dis = -1 means the two nodes are not reachable from each other
//return max range
int get_distance_matrix(GraphXfer* xfer) {
	std::vector<std::vector<int>>& dis_matrix = xfer->dis_matrix;
	size_t tot_op = xfer->srcOps.size(); //the number of src ops
	for (size_t i = 0; i < tot_op; i++) {
		std::vector<int> tmp(tot_op, -1); //initialize all dis to be -1
		dis_matrix.push_back(tmp);
	}

	// prepare another matrix storing the shortest dist from op to input tensor
	std::vector<std::map<int, int>> dis_op_input; //each vector is a map from input tensor id to the shortest dist
	for (size_t i = 0; i < tot_op; i++) {
		std::map<int, int> tmp;
		dis_op_input.push_back(tmp);
	}

	//calculate the shortest distance between any two src nodes
	for (size_t i = 0; i < tot_op; i++) {
		//get the shortest dis from src op i to another op
		dis_matrix[i][i] = 0; //the dis to itself is 0
		int num_to_check = 1; // the number of ops left to check dis from src op i
		int dst = 0; // the shortest dst in last iteration
		while (num_to_check > 0) {
			num_to_check = 0;
			//while there are some ops left
			for (size_t j = 0; j < tot_op; j++) {
				//printf("finding neighbors    i: %zu    j: %zu\n", i, j);
				if (dis_matrix[i][j] == dst) {
					// we need to update all its neighbors' distance
					OpX* opx_check = xfer->srcOps[j];
					//printf("opx_check is what: %zu\n", op2ind(opx_check, xfer->srcOps));
					for (size_t in_i = 0; in_i < opx_check->inputs.size(); in_i++) { // check all input edges
						TensorX& temTX = opx_check->inputs[in_i]; //tem var TensorX
						if (temTX.op != NULL) {
							//not an input tensor
							size_t op_ind = op2ind(temTX.op, xfer->srcOps);
							if (dis_matrix[i][op_ind] < 0) {
								dis_matrix[i][op_ind] = dst + 1;
								num_to_check++;
								/*if (already_checked_op.find(temTX.op) == already_checked_op.end()) {
								if the OpX corresponding to the input has not been checked
								already_checked_op.insert(temTX.op);
								to_Check_op.push_back(temTX.op);
								new_nCheck_op++;*/
							}
						}
						else {
							// an input tensor
							if (dis_op_input[i].find(temTX.idx) == dis_op_input[i].end()) {
								dis_op_input[i][temTX.idx] = dst + 1;
								num_to_check++;
							}
							/*if (already_checked_tx.find(temTX.idx) == already_checked_tx.end()) {
							already_checked_tx.insert(temTX.idx);
							to_Check_tx.push_back(temTX.idx);
							new_nCheck_tx++;
							}*/
						}
					}
					// we also need to check all out edges
					// since the op of the output tensors of an op is itself, we need to iterate over all ops to check the output edges of this op
					for (size_t opx_i = 0; opx_i < tot_op; opx_i++)
					{
						OpX* opx_out = xfer->srcOps[opx_i];
						for (size_t in_i = 0; in_i < (opx_out->inputs.size()); in_i++) {
							TensorX& temTX = opx_out->inputs[in_i];
							if (temTX.op == opx_check) {
								if (dis_matrix[i][opx_i] < 0) {
									dis_matrix[i][opx_i] = dst + 1;
									num_to_check++;
								}
								break;
							}
						}
					}

					//for (size_t out_i = 0; out_i < opx_check->outputs.size(); out_i++) {
					//	printf("checking out edges\n");
					//	TensorX& temTX = opx_check->outputs[out_i]; //tem var TensorX
					//	if (temTX.op != NULL) {
					//		//not a input tensor
					//		size_t op_ind = op2ind(temTX.op, xfer->srcOps);
					//		printf("srcop: %zu   which out edge dst op: %zu\n", i, op_ind);
					//		if (dis_matrix[i][op_ind] < 0) {
					//			dis_matrix[i][op_ind] = dst + 1;
					//			num_to_check++;
					//		}
					//		//if (already_checked_op.find(temTX.op) == already_checked_op.end()) {
					//		//	//if the OpX corresponding to the input has not been checked
					//		//	already_checked_op.insert(temTX.op);
					//		//	to_Check_op.push_back(temTX.op);
					//		//	new_nCheck_op++;
					//		//}
					//	}
					//	else {
					//		//an input tensor
					//		if (dis_op_input[i].find(temTX.idx) == dis_op_input[i].end()) {
					//			dis_op_input[i][temTX.idx] = dst + 1;
					//			num_to_check++;
					//		}
					//		/*if (already_checked_tx.find(temTX.idx) == already_checked_tx.end()) {
					//			already_checked_tx.insert(temTX.idx);
					//			to_Check_tx.push_back(temTX.idx);
					//			new_nCheck_tx++;
					//		}*/
					//	}
					//}
				}
			}
			// we also need to check input tensors
			std::map<int, int>::const_iterator input_it;
			for (input_it = dis_op_input[i].begin(); input_it != dis_op_input[i].end(); input_it++) {
				if (input_it->second == dst) {
					int idx_check = input_it->first;
					for (size_t opx_i = 0; opx_i < tot_op; opx_i++)
					{
						OpX* opx_check = xfer->srcOps[opx_i];
						for (size_t in_i = 0; in_i < (opx_check->inputs.size()); in_i++) {
							TensorX& temTX = opx_check->inputs[in_i];
							if (temTX.op == NULL && temTX.idx == idx_check) {
								if (dis_matrix[i][opx_i] < 0) {
									dis_matrix[i][opx_i] = dst + 1;
									num_to_check++;
								}
								break;
								/*if (already_checked_op.find(opx_check) == already_checked_op.end()) {
								already_checked_op.insert(opx_check);
								to_Check_op.push_back(opx_check);
								new_nCheck_op++;
								}
								break;*/
							}
						}
					}
				}
			}

			dst++;
		}
	}

	//do some check here
	int max_range = -1;
	for (size_t i = 0; i < tot_op; i++) {
		for (size_t j = 0; j < tot_op; j++) {
			//printf("i: %zu   j: %zu   dis[ij]: %d   dis[ji]: %d\n", i, j, dis_matrix[i][j], dis_matrix[j][i]);
			assert(dis_matrix[i][j] == dis_matrix[j][i]);
			if (dis_matrix[i][j] > max_range)
				max_range = dis_matrix[i][j];
			if (dis_matrix[i][j] < 0)
				printf("find a disconnected source graph\n");
		}
	}
	return max_range;
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//calculate affected range for each new op
void get_affected_range(Graph* subGraph, int max_range, std::vector<std::vector<std::set<Op, OpCompare>>>& affected_ranges) {
	//find affected range, store as vector<set<nodes>>

	//std::vector<std::vector<Op>> single_affected_ranges(max_range + 1); //every vector member stores ops between two ranges (circles)
	//std::vector<std::set<Op, OpCompare>> affected_ranges(max_range + 1); //every vector member stores ops in a certain range (circle)

	//printf("finding affected range\n");
	if (subGraph->subst_history.size() == 0)
		return;
	const Graph::GraphSubst& last_step = subGraph->subst_history.back();
	/////////////////
	/*for (size_t dstop_i = 0; dstop_i < last_step.dstOps.size(); dstop_i++) {
	subGraph->inEdges.at(last_step.dstOps[dstop_i]);
	printf("dstop id: %zu    guid: %zu\n", dstop_i, last_step.dstOps[dstop_i].guid);
	}*/
	/////////////////
	for (int range = 0; range <= max_range; range++) {
		//printf("finding range:    %d\n", range);
		if (range == 0) {
			//printf("strange1\n");
			//GraphSubst has the attribute: std::vector<Op> srcOps, dstOps;
			for (size_t dstop_i = 0; dstop_i < last_step.dstOps.size(); dstop_i++) {
				//printf("strange2\n");
				std::vector<std::set<Op, OpCompare>> tmp;
				affected_ranges.push_back(tmp);
				std::set<Op, OpCompare> subtmp;
				affected_ranges[dstop_i].push_back(subtmp);
				/*printf("strange\n");
				printf("size of affected_ranges: %zu\n", affected_ranges.size());
				printf("size of affected_ranges of op %zu: %zu\n", dstop_i, affected_ranges[dstop_i].size());
				printf("size of affected_ranges of op %zu of range %d: %zu\n", dstop_i, range, affected_ranges[dstop_i][range].size());*/
				affected_ranges[dstop_i][range].insert(last_step.dstOps[dstop_i]);
				/*single_affected_ranges[range].push_back(last_step.dstOps[dstop_i]);
				affected_ranges[range].insert(last_step.dstOps[dstop_i]);*/
			}
		}
		else {
			for (size_t dstop_i = 0; dstop_i < last_step.dstOps.size(); dstop_i++) {
				//printf("dstop_i:   %zu\n", dstop_i);
				//printf("range-1: %d  size of affected_ranges of op %zu ---------  last_range: %zu\n", range-1, dstop_i, last_range.size());
				//printf("size of affected_ranges of op %zu of range %d: %zu\n", dstop_i, range - 1, affected_ranges[dstop_i][range - 1].size());
				std::set<Op, OpCompare> subtmp;
				affected_ranges[dstop_i].push_back(subtmp);
				//printf("size of affected_ranges of op %zu: %zu\n", dstop_i, affected_ranges[dstop_i].size());
				//printf("size of affected_ranges of op %zu of range 0: %zu, size of last_range: %zu\n", dstop_i, affected_ranges[dstop_i][0].size(), last_range.size());
				//printf("guid: %zu\n", last_range.begin()->guid);

				affected_ranges[dstop_i][range] = affected_ranges[dstop_i][range - 1];
				//const std::set<Op, OpCompare>& last_range = affected_ranges[dstop_i][range - 1];
				std::set<Op, OpCompare>::const_iterator lastop_it;
				for (lastop_it = affected_ranges[dstop_i][range - 1].begin(); lastop_it != affected_ranges[dstop_i][range - 1].end(); lastop_it++) {
					//check in_edges
					/*printf("checking in edge 1\n");
					printf("guid: %zu\n", lastop_it->guid);
					printf("checking in edge 2\n");*/
					const std::set<Edge, EdgeCompare>& lastop_in = subGraph->inEdges.at(*lastop_it); //calling at() should not throw an execption, because it will always be found
																									 //printf("checking in edge 3\n");
					std::set<Edge, EdgeCompare>::const_iterator in_it;
					for (in_it = lastop_in.begin(); in_it != lastop_in.end(); in_it++) {
						if (in_it->srcOp.ptr != NULL) {
							affected_ranges[dstop_i][range].insert(in_it->srcOp);
						}

						//only normal operator or wrappered inputs/weights can be added into the set
						/*if ((in_it->srcOp.ptr != NULL) && (affected_ranges[range].find(in_it->srcOp) == affected_ranges[range].end())) {
						affected_ranges[range].insert(in_it->srcOp);
						single_affected_ranges[range].push_back(in_it->srcOp);
						}*/
					}
					//printf("checking out edges\n");
					//check out_edges
					std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator lastop_out_it = subGraph->outEdges.find(*lastop_it);
					if (lastop_out_it != subGraph->outEdges.end()) {
						//if this op has outEdges
						const std::set<Edge, EdgeCompare>& lastop_out = lastop_out_it->second;
						std::set<Edge, EdgeCompare>::const_iterator out_it;
						for (out_it = lastop_out.begin(); out_it != lastop_out.end(); out_it++) {
							affected_ranges[dstop_i][range].insert(out_it->dstOp);

							//if (affected_ranges[range].find(out_it->dstOp) == affected_ranges[range].end()) {
							//	//do not need check op.ptr != NULL here, because dstOp can not be null
							//	affected_ranges[range].insert(out_it->dstOp);
							//	single_affected_ranges[range].push_back(out_it->dstOp);
							//}
						}
					}
				}
			}
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
//calculate affected range for each new op
//THIS FUNCTION REQUIRES INPUT OF ORIGINAL OPS WHICH ARE OF DISTANCE 0
void get_affected_range(const std::vector<Op>& ori_ops, Graph* subGraph, int max_range, std::vector<std::vector<std::set<Op, OpCompare>>>& affected_ranges) {

	//const Graph::GraphSubst& last_step = subGraph->subst_history.back();

	for (int range = 0; range <= max_range; range++) {
		//printf("finding range:    %d\n", range);
		if (range == 0) {
			//printf("strange1\n");
			//GraphSubst has the attribute: std::vector<Op> srcOps, dstOps;
			for (size_t dstop_i = 0; dstop_i < ori_ops.size(); dstop_i++) {
				//printf("strange2\n");
				std::vector<std::set<Op, OpCompare>> tmp;
				affected_ranges.push_back(tmp);
				std::set<Op, OpCompare> subtmp;
				affected_ranges[dstop_i].push_back(subtmp);
				/*printf("strange\n");
				printf("size of affected_ranges: %zu\n", affected_ranges.size());
				printf("size of affected_ranges of op %zu: %zu\n", dstop_i, affected_ranges[dstop_i].size());
				printf("size of affected_ranges of op %zu of range %d: %zu\n", dstop_i, range, affected_ranges[dstop_i][range].size());*/
				affected_ranges[dstop_i][range].insert(ori_ops[dstop_i]);
				/*single_affected_ranges[range].push_back(last_step.dstOps[dstop_i]);
				affected_ranges[range].insert(last_step.dstOps[dstop_i]);*/
			}
		}
		else {
			for (size_t dstop_i = 0; dstop_i < ori_ops.size(); dstop_i++) {
				//printf("dstop_i:   %zu\n", dstop_i);
				//printf("range-1: %d  size of affected_ranges of op %zu ---------  last_range: %zu\n", range-1, dstop_i, last_range.size());
				//printf("size of affected_ranges of op %zu of range %d: %zu\n", dstop_i, range - 1, affected_ranges[dstop_i][range - 1].size());
				std::set<Op, OpCompare> subtmp;
				affected_ranges[dstop_i].push_back(subtmp);
				//printf("size of affected_ranges of op %zu: %zu\n", dstop_i, affected_ranges[dstop_i].size());
				//printf("size of affected_ranges of op %zu of range 0: %zu, size of last_range: %zu\n", dstop_i, affected_ranges[dstop_i][0].size(), last_range.size());
				//printf("guid: %zu\n", last_range.begin()->guid);

				affected_ranges[dstop_i][range] = affected_ranges[dstop_i][range - 1];
				//const std::set<Op, OpCompare>& last_range = affected_ranges[dstop_i][range - 1];
				std::set<Op, OpCompare>::const_iterator lastop_it;
				for (lastop_it = affected_ranges[dstop_i][range - 1].begin(); lastop_it != affected_ranges[dstop_i][range - 1].end(); lastop_it++) {
					//check in_edges
					/*printf("checking in edge 1\n");
					printf("guid: %zu\n", lastop_it->guid);
					printf("checking in edge 2\n");*/
					const std::set<Edge, EdgeCompare>& lastop_in = subGraph->inEdges.at(*lastop_it); //calling at() should not throw an execption, because it will always be found
																									 //printf("checking in edge 3\n");
					std::set<Edge, EdgeCompare>::const_iterator in_it;
					for (in_it = lastop_in.begin(); in_it != lastop_in.end(); in_it++) {
						if (in_it->srcOp.ptr != NULL) {
							affected_ranges[dstop_i][range].insert(in_it->srcOp);
						}

						//only normal operator or wrappered inputs/weights can be added into the set
						/*if ((in_it->srcOp.ptr != NULL) && (affected_ranges[range].find(in_it->srcOp) == affected_ranges[range].end())) {
						affected_ranges[range].insert(in_it->srcOp);
						single_affected_ranges[range].push_back(in_it->srcOp);
						}*/
					}
					//printf("checking out edges\n");
					//check out_edges
					std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator lastop_out_it = subGraph->outEdges.find(*lastop_it);
					if (lastop_out_it != subGraph->outEdges.end()) {
						//if this op has outEdges
						const std::set<Edge, EdgeCompare>& lastop_out = lastop_out_it->second;
						std::set<Edge, EdgeCompare>::const_iterator out_it;
						for (out_it = lastop_out.begin(); out_it != lastop_out.end(); out_it++) {
							affected_ranges[dstop_i][range].insert(out_it->dstOp);

							//if (affected_ranges[range].find(out_it->dstOp) == affected_ranges[range].end()) {
							//	//do not need check op.ptr != NULL here, because dstOp can not be null
							//	affected_ranges[range].insert(out_it->dstOp);
							//	single_affected_ranges[range].push_back(out_it->dstOp);
							//}
						}
					}
				}
			}
		}
	}
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
//get the affected range of all NEAR ops
//hop_num is the "near" distance
//max_range is the max distance between two nodes in the subst source graph of the last subst step of subGraph
void get_affected_range_near(std::vector<std::vector<std::set<Op, OpCompare>>>& affected_ranges, Graph* subGraph, int hop_num, std::vector<Op>& the_near_ops, int max_range) {
	// we first find all near ops
	std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_tmp;
	get_affected_range(subGraph, hop_num, affected_ranges_tmp);
	std::set<Op, OpCompare> all_ops;
	for (size_t i = 0; i < affected_ranges_tmp.size(); i++)
	{
		all_ops.insert(affected_ranges_tmp[i][hop_num].begin(), affected_ranges_tmp[i][hop_num].end());
	}
	
	//then we find affected ops of all_ops
	std::set<Op, OpCompare>::const_iterator it;
	for (it = all_ops.begin(); it != all_ops.end(); ++it) {
		the_near_ops.push_back(*it);
	}
	get_affected_range(the_near_ops, subGraph, max_range, affected_ranges);

}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
Graph* Graph::optimize_reuse(float alpha, int budget, bool print_subst)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);


	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//calculate the distance from the first node in a source graph to any other node in the source graph  
	int max_range = -1;
	for (size_t i = 0; i < xfers.size(); i++) {
		//printf("xferrrrrrrrrrrr: %zu\n", i);
		int tmp = get_distance_matrix(xfers[i]);
		if (tmp > max_range)
			max_range = tmp;
	}

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	std::deque<SearchTreeNode> candidates;

	std::set<size_t> hashmap;

	//////////////////////////
	SearchTreeNode rootNode(this, -1);
	//candidates.push_back(this);
	candidates.push_back(rootNode);
	//////////////////////////

	//hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;


	Graph *bestGraph = this;
	float bestCost = total_cost();
	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();


	size_t add_counter = 1; //how many searchTreeNodes in candidates have been pushed
							//size_t del_counter = 0; //how many searchTreeNodes in candidates have been deleted
	size_t reuse_counter = 0;

	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");

	int last_to_search = 0; //store the last node to search, not necessarily the last node in the candidates list
							//the following search procedure is different from the previous one. This one does DFS search to reduce memory consumption
							//while (!candidates.empty())
	while (true) //we do not check empty here because "update_last_to_search" does this
	{
		//first update the last_to_search and the candidates list
		last_to_search = update_last_to_search(candidates, last_to_search, bestGraph, original_graph);
		if (last_to_search == -1) //all nodes has been searched
			break;

		//printf("the last to search node id: %d\n", last_to_search);

		//get the node to search in this round
		SearchTreeNode& this_node = candidates.at(last_to_search);
		this_node.searched = true; //this node has been searched
		Graph* subGraph = this_node.graphPtr;

		//update the best graph
		if (subGraph->total_cost() < bestCost) {
			//delete bestGraph;
			bestCost = subGraph->total_cost();
			bestGraph = subGraph;
		}

		//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
		if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
		{
			/*if (subGraph != bestGraph)
			delete subGraph;*/

			continue;
		}

		// we need to calculate the affected range of each new op of this graph
		std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
		get_affected_range(subGraph, max_range, affected_ranges);

		// try to find the child nodes of this node
		this_node.first_child = candidates.size(); //update the child node range of this node
		if (subGraph->subst_history.size() > 0) {
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < subGraph->subst_history.back().dstOps.size(); newopid++) {

						//printf("DEPTHID: %zu   NEWOPID: %zu\n", depthid, newopid);

						//RECORD THE START OF SUBST
						auto time_s = std::chrono::system_clock::now();

						xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search, gentime);
					
						//RECORD THE END OF SUBST
						auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;
					}
				}
			}
			
			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();

			xfers[0]->reuse_all_steps(candidates, last_to_search, add_counter, reuse_counter, gentime);

			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;
		}
		else {
			//this node has no subst history, then cannot reuse
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				//RECORD THE START OF SUBST
				auto time_s = std::chrono::system_clock::now();

				xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, -1, -1, last_to_search, gentime);
			
				//RECORD THE END OF SUBST
				auto time_e = std::chrono::system_clock::now();
				auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
				double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
				run_function_time += searchtime_in;
			}
		}

		this_node.last_child = candidates.size(); //update the child node range of this node

												  /*if (subGraph != bestGraph)
												  delete subGraph;*/

	}

	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	if (print_subst) {
		printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
			printf("        substitution[%03zu]: \n", i);
			Graph::GraphSubst subst = bestGraph->subst_history[i];
			for (size_t j = 0; j < subst.srcOps.size(); j++) {
				printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
			}
			for (size_t j = 0; j < subst.dstOps.size(); j++) {
				printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
			}
		}
	}

	////////////////
	//print counter
	printf("        add_counter: %zu\n", add_counter);

	printf("        reuse_counter: %zu\n", reuse_counter);

	////////////////
	return bestGraph;
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
// check whether to_check has some common ops with the all_ops set
// returns true if there is any common ops; otherwise false;
bool check_common_ops(const std::set<Op, OpCompare>& all_ops, const Graph::GraphSubst& to_check) {
	const std::vector<Op>& to_check_srcs = to_check.srcOps;
	std::set<int>::const_iterator it;
	for (size_t i = 0; i < to_check_srcs.size(); i++) {

		if (all_ops.find(to_check_srcs[i]) != all_ops.end())
			return true;
	}
	return false;
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
// sample the best step (including those making the cost increase and all the steps near it
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes
void do_sampling_best_local_old(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates, int hop_num) {
	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap;

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;
	
	//first find the best step
	float best_cost = -1;
	size_t best_node_id = -1;

	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;

		if (best_cost < 0) {
			best_cost = tem_graph->total_cost();
			best_node_id = weight_i;
		}
		else {
			if (tem_graph->total_cost() < best_cost) {
				best_cost = tem_graph->total_cost();
				best_node_id = weight_i;
			}
		}
	}
	
	// sample the bset step
	candidates.at(best_node_id).sample_quota = 1;
	hashmap.insert(candidates.at(best_node_id).graphPtr->hash());
	sample_size--;
	
	//find the steps around the best step (1 hop)
	//size_t hop_num = 1;
	std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
	get_affected_range(candidates.at(best_node_id).graphPtr, hop_num, affected_ranges);
	
	std::set<Op, OpCompare> all_near_ops; //store 1 hop ops of all src ops of the best step
	all_near_ops.insert(candidates.at(best_node_id).graphPtr->subst_history.back().srcOps.begin(), candidates.at(best_node_id).graphPtr->subst_history.back().srcOps.end());
	for (size_t i = 0; i < affected_ranges.size(); i++)
	{
		all_near_ops.insert(affected_ranges[i][hop_num].begin(), affected_ranges[i][hop_num].end());
	}


	//now we find the nodes which has at least one of the near ops	
	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;
		if (weight_i == best_node_id)
			continue;
		if (check_common_ops(all_near_ops, tem_graph->subst_history.back())) {
			// if is near enough
			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			//add_to_uniq_topk(topknodes_good, (size_t)sample_size, &(candidates.at(weight_i)));
		
			if (tem_graph->subst_history.back().cost_change > 0)
				add_to_uniq_topk(topknodes_bad, (size_t)(sample_size / 2), &(candidates.at(weight_i)));
			else
				add_to_uniq_topk(topknodes_good, (size_t)(sample_size / 2), &(candidates.at(weight_i)));
		}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
//this is the subroutine of do_sampling_best_local: select the steps near the best step (best_node_id)
void sub_sample_best_local(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates, int hop_num, size_t best_node_id) {
	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap;

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;
	
	// sample the bset step
	candidates.at(best_node_id).sample_quota = 1;
	hashmap.insert(candidates.at(best_node_id).graphPtr->hash());
	sample_size--;

	//find the steps around the best step (1 hop)
	//size_t hop_num = 1;
	std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
	get_affected_range(candidates.at(best_node_id).graphPtr, hop_num, affected_ranges);

	std::set<Op, OpCompare> all_near_ops; //store 1 hop ops of all src ops of the best step
	all_near_ops.insert(candidates.at(best_node_id).graphPtr->subst_history.back().srcOps.begin(), candidates.at(best_node_id).graphPtr->subst_history.back().srcOps.end());
	for (size_t i = 0; i < affected_ranges.size(); i++)
	{
		all_near_ops.insert(affected_ranges[i][hop_num].begin(), affected_ranges[i][hop_num].end());
	}


	//now we find the nodes which has at least one of the near ops	
	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;
		if (weight_i == best_node_id)
			continue;
		if (check_common_ops(all_near_ops, tem_graph->subst_history.back())) {
			// if is near enough
			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			//add_to_uniq_topk(topknodes_good, (size_t)sample_size, &(candidates.at(weight_i)));
			if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size() - 2).cost_change > 0))
			//if (tem_graph->subst_history.back().cost_change > 0)
				add_to_uniq_topk(topknodes_bad, (size_t)(sample_size / 2), &(candidates.at(weight_i)));
			else
				add_to_uniq_topk(topknodes_good, (size_t)(sample_size / 2), &(candidates.at(weight_i)));
		}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
// sample the best step (including those making the cost increase and all the steps near it
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes
void do_sampling_best_local(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates, int hop_num) {
	//first find the best step
	float best_cost = -1;
	size_t best_node_id = -1;

	float best_cost_bad = -1;
	size_t best_node_id_bad = -1;

	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;

		if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size() - 2).cost_change > 0)) {
			if (best_cost_bad < 0) {
				best_cost_bad = tem_graph->total_cost();
				best_node_id_bad = weight_i;
			}
			else {
				if (tem_graph->total_cost() < best_cost_bad) {
					best_cost_bad = tem_graph->total_cost();
					best_node_id_bad = weight_i;
				}
			}
		}
		else {
			if (best_cost < 0) {
				best_cost = tem_graph->total_cost();
				best_node_id = weight_i;
			}
			else {
				if (tem_graph->total_cost() < best_cost) {
					best_cost = tem_graph->total_cost();
					best_node_id = weight_i;
				}
			}
		}
	}

	// do sampling
	sub_sample_best_local(level_end, del_counter, (int)(sample_size / 2), candidates, hop_num, best_node_id);
	sub_sample_best_local(level_end, del_counter, (int)(sample_size / 2), candidates, hop_num, best_node_id_bad);

}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
//This function assigns the sample quota to every candidate in the new level
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes OF TYPE SearchTreeNode
void do_sampling_old(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {
	//std::vector<float> tem_sample_weights;
	//printf("----------start sample---------------\n");

	///////////////Use the weight of each node as the probability of it to be sampled
	//below is to find the biggest cost reduction
	//float biggest_cost_reduction = 0;
	//int num_hopeless = 0;
	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap;

	//printf("\n DO NORMAL SAMPLING           level end: %zu    del counter: %zu\n", level_end, del_counter);

	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;

	//look ahead for one step
	if (level_end == 0) {
		//this is the first level, no need to check last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			//if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			//else {
			if (tem_graph->subst_history.back().cost_change > 0)
				add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//}
		}
	}
	else {
		//printf("----------------------sampling 2\n");

		//need to check the last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				continue;
			}


			if (tem_graph->subst_history.back().cost_change > 0) {
				//int last_depend_step = tem_graph->subst_history.back().biggestNode.SIStepOrder;
				int last_step = tem_graph->subst_history.size() - 2;
				/*if (last_depend_step == -1)
				candidates.at(weight_i).sample_quota = 1;*/
				if (tem_graph->subst_history.at(last_step).cost_change < 0) {
					//if the cost does not increase and increase for twice
					add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
					//candidates.at(weight_i).sample_quota = 1;
				}
				else
					num_bad_bad++;
			}
			//else if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//candidates.at(weight_i).sample_quota = 1;
		}
	}

	//printf("good num: %zu        bad num: %zu\n", topknodes_good.size(), topknodes_bad.size());

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION AND WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes OF TYPE SearchTreeNode
//THIS FUNCTION ASSUMES IF THERE IS A STEP INCREASING COST, WE WOULD IMMEDIATELY FIND ANOTHER STEP DEPENDENT ON IT
void do_sampling_two_increases(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	//int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap;

	//printf("\n DO NORMAL SAMPLING           level end: %zu    del counter: %zu\n", level_end, del_counter);

	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;


	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;
		tot_node++;

		if (tem_graph->subst_history.back().cost_change > 0)
			continue;

		//if last step increases the cost, the next step must depend on that step
		if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size() - 2).cost_change > 0)) {
			Graph::GraphSubst& lastlast = tem_graph->subst_history.at(tem_graph->subst_history.size() - 2);
			std::set<Op, OpCompare> all_ops(lastlast.dstOps.begin(), lastlast.dstOps.end());
			if (check_common_ops(all_ops, tem_graph->subst_history.back()) == false)
				continue;
		}


		if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//printf("add candidate!\n");
			hashmap.insert(tem_graph->hash());
		}
		else {
			continue;
		}

		//if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
		//  num_hopeless++;
		//else {
		if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size()-2).cost_change > 0))
			add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		else
			add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		//}
	}

	
	//printf("good num: %zu        bad num: %zu\n", topknodes_good.size(), topknodes_bad.size());

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION AND WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes OF TYPE SearchTreeNode
//THIS FUNCTION ASSUMES IF THERE IS A STEP INCREASING COST, WE WOULD IMMEDIATELY FIND ANOTHER STEP DEPENDENT ON IT
//ALL SUBST STEPS FOUND AFTER COST-up substitutions ARE DEPENDENT ON THEM
void do_sampling_two_increases_nocheck(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	//int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap;

	//printf("\n DO NORMAL SAMPLING           level end: %zu    del counter: %zu\n", level_end, del_counter);

	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;


	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;
		tot_node++;

		if (tem_graph->subst_history.back().cost_change > 0)
			continue;

		//if last step increases the cost, the next step must depend on that step
		/*if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size() - 2).cost_change > 0)) {
			Graph::GraphSubst& lastlast = tem_graph->subst_history.at(tem_graph->subst_history.size() - 2);
			std::set<Op, OpCompare> all_ops(lastlast.dstOps.begin(), lastlast.dstOps.end());
			if (check_common_ops(all_ops, tem_graph->subst_history.back()) == false)
				continue;
		}*/


		if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//printf("add candidate!\n");
			hashmap.insert(tem_graph->hash());
		}
		else {
			continue;
		}

		//if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
		//  num_hopeless++;
		//else {
		if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size() - 2).cost_change > 0))
			add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		else
			add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		//}
	}


	//printf("good num: %zu        bad num: %zu\n", topknodes_good.size(), topknodes_bad.size());

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION AND WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes OF TYPE SearchTreeNode
//THIS FUNCTION ASSUMES IF THERE IS A STEP INCREASING COST, WE WOULD IMMEDIATELY FIND ANOTHER STEP DEPENDENT ON IT
//ALL SUBST STEPS FOUND AFTER COST-up substitutions ARE DEPENDENT ON THEM
void do_sampling_two_increases_nocheck_2hash(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	//int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap_up;
	std::set<size_t> hashmap_down;

	//printf("\n DO NORMAL SAMPLING           level end: %zu    del counter: %zu\n", level_end, del_counter);

	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;


	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;
		tot_node++;

		if (tem_graph->subst_history.back().cost_change > 0)
			continue;

		//if last step increases the cost, the next step must depend on that step
		/*if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size() - 2).cost_change > 0)) {
		Graph::GraphSubst& lastlast = tem_graph->subst_history.at(tem_graph->subst_history.size() - 2);
		std::set<Op, OpCompare> all_ops(lastlast.dstOps.begin(), lastlast.dstOps.end());
		if (check_common_ops(all_ops, tem_graph->subst_history.back()) == false)
		continue;
		}*/

		//if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
		//  num_hopeless++;
		//else {
		if ((tem_graph->subst_history.size() > 1) && (tem_graph->subst_history.at(tem_graph->subst_history.size() - 2).cost_change > 0)) {

			if (hashmap_up.find(tem_graph->hash()) == hashmap_up.end()) {
				//printf("add candidate!\n");
				hashmap_up.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		}	
		else {

			if (hashmap_down.find(tem_graph->hash()) == hashmap_down.end()) {
				//printf("add candidate!\n");
				hashmap_down.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		}
			
		//}
	}


	//printf("good num: %zu        bad num: %zu\n", topknodes_good.size(), topknodes_bad.size());

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION AND WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// level_end: the end of the last level
// del_counter: the number of deleted nodes in the candidate list
// sample_size: the sample size of the sampletrick search algo
// candidates: the candidate list of all current existing nodes OF TYPE SearchTreeNode
void do_sampling(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {
	
	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap;



	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;

	//look ahead for one step
	if (level_end == 0) {
		//this is the first level, no need to check last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			if (tem_graph->subst_history.back().cost_change > 0)
				add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//}
		}
	}
	else {
		//need to check the last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(tem_graph->hash());
			}
			else {
				//print the substitution history
				/*printf("FAILED IN HASHMAP CHECKING\n");
				print_subst_history_n_cost(tem_graph);*/

				continue;
			}


			if (tem_graph->subst_history.back().cost_change > 0) {
				//int last_depend_step = tem_graph->subst_history.back().biggestNode.SIStepOrder;
				int last_step = tem_graph->subst_history.size() - 2;
				/*if (last_depend_step == -1)
				candidates.at(weight_i).sample_quota = 1;*/
				if (tem_graph->subst_history.at(last_step).cost_change < 0) {
					//if the cost does not increase and increase for twice
					add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
					//candidates.at(weight_i).sample_quota = 1;
				}
				else
					num_bad_bad++;
			}
			//else if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//candidates.at(weight_i).sample_quota = 1;
		}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION AND WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// WITH NO HASH MAP
void do_sampling_nohash(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	//std::set<size_t> hashmap;



	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;

	//look ahead for one step
	if (level_end == 0) {
		//this is the first level, no need to check last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}

			if (tem_graph->subst_history.back().cost_change > 0)
				add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//}
		}
	}
	else {
		//need to check the last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}


			if (tem_graph->subst_history.back().cost_change > 0) {
				//int last_depend_step = tem_graph->subst_history.back().biggestNode.SIStepOrder;
				int last_step = tem_graph->subst_history.size() - 2;
				/*if (last_depend_step == -1)
				candidates.at(weight_i).sample_quota = 1;*/
				if (tem_graph->subst_history.at(last_step).cost_change < 0) {
					//if the cost does not increase and increase for twice
					add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
					//candidates.at(weight_i).sample_quota = 1;
				}
				else
					num_bad_bad++;
			}
			//else if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			else
				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			//candidates.at(weight_i).sample_quota = 1;
		}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION AND WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// TWO HASHMAP FOR COST-INCREASING SUBSTITUTION AND COST-DECREASING SUBSTITUTION RESPECTIVELY
void do_sampling_2independent_hash(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap_up;
	std::set<size_t> hashmap_down;



	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;

	//look ahead for one step
	if (level_end == 0) {
		//this is the first level, no need to check last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}

			if (tem_graph->subst_history.back().cost_change > 0) {
				if (hashmap_up.find(tem_graph->hash()) == hashmap_up.end()) {
					//printf("add candidate!\n");
					hashmap_up.insert(tem_graph->hash());
				}
				else {
					continue;
				}

				add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			}
			else {
				if (hashmap_down.find(tem_graph->hash()) == hashmap_down.end()) {
					//printf("add candidate!\n");
					hashmap_down.insert(tem_graph->hash());
				}
				else {
					continue;
				}

				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			}
			//}
		}
	}
	else {
		//need to check the last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}


			if (tem_graph->subst_history.back().cost_change > 0) {
				//int last_depend_step = tem_graph->subst_history.back().biggestNode.SIStepOrder;
				int last_step = tem_graph->subst_history.size() - 2;
				/*if (last_depend_step == -1)
				candidates.at(weight_i).sample_quota = 1;*/
				if (tem_graph->subst_history.at(last_step).cost_change < 0) {
					
					if (hashmap_up.find(tem_graph->hash()) == hashmap_up.end()) {
						//printf("add candidate!\n");
						hashmap_up.insert(tem_graph->hash());
					}
					else {
						continue;
					}

					//if the cost does not increase and increase for twice
					add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
					//candidates.at(weight_i).sample_quota = 1;
				}
				else
					num_bad_bad++;
			}
			//else if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			else {
				if (hashmap_down.find(tem_graph->hash()) == hashmap_down.end()) {
					//printf("add candidate!\n");
					hashmap_down.insert(tem_graph->hash());
				}
				else {
					continue;
				}

				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			}
				
			//candidates.at(weight_i).sample_quota = 1;
		}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION AND WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// TWO HASHMAP FOR COST-INCREASING SUBSTITUTION AND COST-DECREASING SUBSTITUTION RESPECTIVELY
// FIND COST-DECREASING SUBSTITUTION FIRST, COST-INCREASING SUBSTITUTIONS ARE NOT THE SAME AS COST-DECREASING ONES
void do_sampling_2dependent_hash(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap_up;
	std::set<size_t> hashmap_down;



	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;

	//look ahead for one step
	if (level_end == 0) {
		//this is the first level, no need to check last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}

			if (tem_graph->subst_history.back().cost_change > 0) {
				continue;
			}
			else {
				if (hashmap_down.find(tem_graph->hash()) == hashmap_down.end()) {
					//printf("add candidate!\n");
					hashmap_down.insert(tem_graph->hash());
				}
				else {
					continue;
				}

				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			}
			//}
		}
	}
	else {
		//need to check the last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}


			if (tem_graph->subst_history.back().cost_change > 0) {
				continue;
			}
			//else if ((tem_graph->total_cost() - best_step_cost_reduc * (budget - tem_graph->subst_history.size())) > bestCost_for_sample)
			//  num_hopeless++;
			else {
				if (hashmap_down.find(tem_graph->hash()) == hashmap_down.end()) {
					//printf("add candidate!\n");
					hashmap_down.insert(tem_graph->hash());
				}
				else {
					continue;
				}

				add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			}

			//candidates.at(weight_i).sample_quota = 1;
		}
	}

	while (!topknodes_good.empty()) {
		//store the hash value of selected good ones
		hashmap_up.insert(topknodes_good.top()->graphPtr->hash());

		topknodes_good.top()->sample_quota = 1;
		topknodes_good.pop();
	}

	//look ahead for one step
	if (level_end == 0) {
		//this is the first level, no need to check last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}

			if (tem_graph->subst_history.back().cost_change > 0) {
				if (hashmap_up.find(tem_graph->hash()) == hashmap_up.end()) {
					//printf("add candidate!\n");
					hashmap_up.insert(tem_graph->hash());
				}
				else {
					continue;
				}

				add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
			}
			//}
		}
	}
	else {
		//need to check the last step
		for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
			Graph* tem_graph = candidates.at(weight_i).graphPtr;
			tot_node++;

			//if (hashmap.find(tem_graph->hash()) == hashmap.end()) {
			//	//printf("add candidate!\n");
			//	hashmap.insert(tem_graph->hash());
			//}
			//else {
			//	continue;
			//}


			if (tem_graph->subst_history.back().cost_change > 0) {
				//int last_depend_step = tem_graph->subst_history.back().biggestNode.SIStepOrder;
				int last_step = tem_graph->subst_history.size() - 2;
				/*if (last_depend_step == -1)
				candidates.at(weight_i).sample_quota = 1;*/
				if (tem_graph->subst_history.at(last_step).cost_change < 0) {

					if (hashmap_up.find(tem_graph->hash()) == hashmap_up.end()) {
						//printf("add candidate!\n");
						hashmap_up.insert(tem_graph->hash());
					}
					else {
						continue;
					}

					//if the cost does not increase and increase for twice
					add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
					//candidates.at(weight_i).sample_quota = 1;
				}
				else
					num_bad_bad++;
			}			

			//candidates.at(weight_i).sample_quota = 1;
		}
	}


	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;
		topknodes_bad.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// TWO HASHMAP FOR COST-INCREASING SUBSTITUTION AND COST-DECREASING SUBSTITUTION RESPECTIVELY
// NOT CHECK WHETHER  2 CONSECUTIVE COST-UP STEPS OR NOT
void do_sampling_2independent_hash_nocheck(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	std::set<size_t> hashmap_up;
	std::set<size_t> hashmap_down;



	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodepotentialCompare> topknodes_bad;

	//this is the first level, no need to check last step
	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;
		tot_node++;

		if (tem_graph->subst_history.back().cost_change > 0) {
			if (hashmap_up.find(tem_graph->hash()) == hashmap_up.end()) {
				//printf("add candidate!\n");
				hashmap_up.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			add_to_uniq_topk(topknodes_bad, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		}
		else {
			if (hashmap_down.find(tem_graph->hash()) == hashmap_down.end()) {
				//printf("add candidate!\n");
				hashmap_down.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		}
		//}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;
		
		candidates.at(topknodes_good.top()->father - del_counter).has_sampled_child = true;

		topknodes_good.pop();
	}

	while (!topknodes_bad.empty()) {
		topknodes_bad.top()->sample_quota = 1;

		candidates.at(topknodes_bad.top()->father - del_counter).has_sampled_child = true;
		//topknodes_bad.top()->further_search = true;

		topknodes_bad.pop();
	}
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH NEW REUSE----------------------//
//This function assigns the sample quota to every candidate in the new level
// TWO HASHMAP FOR COST-INCREASING SUBSTITUTION AND COST-DECREASING SUBSTITUTION RESPECTIVELY
// ONLY SAMPLE COST-DOWN SUBSTITUTIONS
void do_sampling_down_dependent(size_t level_end, size_t del_counter, int sample_size, std::deque<SearchTreeNode>& candidates) {

	int num_bad_bad = 0;
	int tot_node = 0;
	//size_t biggest_node_ind = level_end - del_counter + 1;

	//use hashmap here to filter redundant graphs
	//std::set<size_t> hashmap_up;
	std::set<size_t> hashmap_down;



	//use priority queue to select the top k nodes
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_good; //store nodes which decrease cost
	//std::set<SearchTreeNode*, SearchTreeNodeCompare> topknodes_bad; //store nodes which increase cost

	std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_good;
	//std::priority_queue<SearchTreeNode*, std::vector<SearchTreeNode*>, SearchTreeNodeCompare> topknodes_bad;

	//this is the first level, no need to check last step
	for (size_t weight_i = level_end - del_counter + 1; weight_i < candidates.size(); weight_i++) {
		Graph* tem_graph = candidates.at(weight_i).graphPtr;
		tot_node++;

		if (tem_graph->subst_history.back().cost_change <= 0) {
			//we only find dependent nodes, which is not reused
			if (candidates.at(weight_i).reused == true)
				continue;

			if (hashmap_down.find(tem_graph->hash()) == hashmap_down.end()) {
				//printf("add candidate!\n");
				hashmap_down.insert(tem_graph->hash());
			}
			else {
				continue;
			}

			add_to_uniq_topk(topknodes_good, (size_t)sample_size / 2, &(candidates.at(weight_i)));
		}
		//}
	}

	while (!topknodes_good.empty()) {
		topknodes_good.top()->sample_quota = 1;

		candidates.at(topknodes_good.top()->father - del_counter).has_sampled_child = true;

		topknodes_good.pop();
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
//this function gets the best cost change among all nodes in candidates
float get_best_change(const std::deque<SearchTreeNode>& candidates) {
	if (candidates.empty())
		return 100; 
	float best_change = candidates[0].graphPtr->subst_history.back().cost_change;
	for (size_t i = 1; i < candidates.size(); i++) {
		if (candidates[i].graphPtr->subst_history.back().cost_change < best_change)
			best_change = candidates[i].graphPtr->subst_history.back().cost_change;
	}
	return best_change;
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION----------------------//
Graph* Graph::optimize_sampletrick_local(float alpha, int budget, bool print_subst, int sample_size)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);



	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//calculate the distance from a node in the source graph of a rule to another node in the source graph
	int max_range = -1;
	for (size_t i = 0; i < xfers.size(); i++) {
		//printf("xferrrrrrrrrrrr: %zu\n", i);
		int tmp = get_distance_matrix(xfers[i]);
		if (tmp > max_range)
			max_range = tmp;
	}

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<SearchTreeNodesample> candidates;
	std::deque<SearchTreeNode> candidates;

	std::vector<std::set<size_t>> hashmaps; //assign a hashmap for each level

	//////////////////////////
	
	//SearchTreeNodesample rootNode(this);
	SearchTreeNode rootNode(this, -1);
	//candidates.push_back(this);
	//store the sample quota
	rootNode.sample_quota = (size_t)sample_size;
	candidates.push_back(rootNode);
	//////////////////////////

	//hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;


	Graph *bestGraph = this;
	//float bestCost = total_cost();
	//below variable records the bestCost which is updated every time a node is finding its child nodes
	float bestCost_for_sample = total_cost();
	//float best_step_cost_reduc = 0; // the largest cost reduction by one step currently

									//printf("MetaFlow Cost = %.4lfms\n", bestCost);
									//printf("Input graph: end-to-end execution time =\n"
									//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();


	size_t add_counter = 1; //how many searchTreeNodes in candidates have been pushed
	size_t del_counter = 0; //how many searchTreeNodes in candidates have been deleted
	size_t reuse_counter = 0;

	int hop_num = 1; //only find steps which are 1 hop away

	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");


	//assign a hashmap set for level 1
	std::set<size_t> hashmap_level_1;
	hashmaps.push_back(hashmap_level_1);


	//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_tmp;
	//store the explored sequences as a tree in a deque
	for (size_t i = 0; i < xfers.size(); i++) {
		std::vector<std::set<Op, OpCompare>> affected_ranges_tmp;
		Op op_tmp;
		std::set<Op, OpCompare> near_ops_tmp;
		//std::set<Op, OpCompare> empty_range;
		//candidates[0].childrenNode.push_back(std::vector<size_t>());
		//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
		//xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, add_counter, -1, -1, 0);
		
		//RECORD THE START OF SUBST
		auto time_s = std::chrono::system_clock::now();

		xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, op_tmp, near_ops_tmp, add_counter, -1, 0, gentime);
	
		//RECORD THE END OF SUBST
		auto time_e = std::chrono::system_clock::now();
		auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
		double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		run_function_time += searchtime_in;
	}

	bool sample_best_local = true; //if true, sample the best step and the steps near it


	///////////////////////////////CODE for re-allocating sample quota
	//size_t level_start = 0; //the original index of the start node in a level
	size_t level_end = 0; //the original index of the end node in a level
						  ///////////////////////////////END CODE for re-allocating sample quota

						  ///////////////////////CODE record the cost range of the sampled nodes in each level
	/*std::vector<float> upperbound;
	std::vector<float> lowerbound;*/
	///////////////////////END CODE

	//we first pop the first node (with original graph)
	candidates.pop_front();
	del_counter++;

	int level_num = 1;

	float last_level_best_change = -1;
	float this_level_best_change = get_best_change(candidates);

	//rewrite the loop structure for this algo: sample trick local version
	while (!candidates.empty())
	{
		if (level_num >= 2 * budget)
			break;

		// sample the nodes of this level
		//printf("\n=============NEW LEVEL %d=============\n", level_num);

		// if the last step is to increase the cost, we need to directly find the next step which depends on it
		size_t level_size = candidates.size(); //the total number of nodes in this level
		for (size_t i = 0; i < level_size; i++) {
			size_t tem_pos = i;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			if (subGraph->subst_history.back().cost_change <= 0) {
				continue;
			}

			//print whether this graph is sampled or not
			//printf("\n---------------SELF: %zu    FURTHER SEARCH-----------", del_counter + i);// , candidates.at(tem_pos).sample_quota);
			//print subst history infor for each graph, no matter whether it is sampled or nor
			//print_subst_history_n_cost(subGraph);

			//candidates.pop_front();

			//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				continue;
			}


			//find all near ops and affected range of them
			std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			std::vector<Op> the_near_ops;
			get_affected_range_near(affected_ranges, subGraph, 0, the_near_ops, max_range);
			std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			//SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			// try to find the child nodes of this node which are NEAR
			//this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < the_near_ops.size(); newopid++) {
						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);
						
						//RECORD THE START OF SUBST
						auto time_s = std::chrono::system_clock::now();
						
						xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, tem_pos, gentime);
					
						//RECORD THE END OF SUBST
						auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;
					}
				}
			}
		}


		if (sample_best_local)
			do_sampling_best_local(level_end, del_counter, sample_size, candidates, hop_num);
		else
			do_sampling_two_increases(level_end, del_counter, sample_size, candidates);


		//update level_start and level_end
		//level_start = level_end + 1;
		level_end = candidates.size() - 1 + del_counter;

		//assign a hashmap set for the next level to be searched
		std::set<size_t> hashmap_level_next;
		hashmaps.push_back(hashmap_level_next);

		//update last_level_best_change
		last_level_best_change = this_level_best_change;

		//then we search child nodes of all nodes in this level
		level_size = candidates.size(); //the total number of nodes in this level
		for (size_t i = 0; i < level_size; i++) {
			size_t tem_pos = 0;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			//print whether this graph is sampled or not
			//printf("\n---------------SELF: %zu    SAMPLE QUOTA: %zu-----------", del_counter, candidates.at(tem_pos).sample_quota);
			//print subst history infor for each graph, no matter whether it is sampled or nor
			//print_subst_history_n_cost(subGraph);

			//update bestCost_for_sample
			if (subGraph->total_cost() < bestCost_for_sample) {
				//delete bestGraph;
				delete_graph(original_graph, bestGraph);

				bestCost_for_sample = subGraph->total_cost();
				bestGraph = subGraph;

			}

			if (candidates.at(tem_pos).sample_quota == 0) {
				//there is no quota for this child node, i.e., it is not sampled

				if (subGraph != bestGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}

				candidates.pop_front();
				del_counter++;

				continue;
			}
			//////////////////////////////////////////////////////////////////////

			//candidates.pop_front();

			//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				if (subGraph != bestGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}

				candidates.pop_front();
				del_counter++;

				continue;
			}


			//find all near ops and affected range of them
			std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			std::vector<Op> the_near_ops;
			get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
			std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			//SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			// try to find the child nodes of this node which are NEAR
			//this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < the_near_ops.size(); newopid++) {
						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);
						
						//RECORD THE START OF SUBST
						auto time_s = std::chrono::system_clock::now();
						
						xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, tem_pos, gentime);
					
						//RECORD THE END OF SUBST
						auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;
					}
				}
			}
			//xfers[0]->reuse_all_steps(candidates, last_to_search, add_counter, reuse_counter);

			//this_node.last_child = candidates.size() + del_counter; //update the child node range of this node


																	//delete the subGraph if it is not the best
			if (subGraph != bestGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}

			candidates.pop_front();
			del_counter++;
		}
	
		//update this_level_best_change
		this_level_best_change = get_best_change(candidates);

		//here all nodes of last level must have been deleted
		if ((candidates.empty() && (bestGraph->subst_history.size() < (size_t)budget)) || (level_num%2==0) ) {//( (!candidates.empty()) && (this_level_best_change > 0.4 * last_level_best_change) )) {
			//printf("\n====================NEED SEARCH GLOBALLY=====================  candidates empty? %d   last max: %.4lf   this max: %.4lf\n", (candidates.empty()?1:0), last_level_best_change, this_level_best_change);
			//if the no near steps can be found in the previous process
			sample_best_local = true;
			//we need to go back to the current best graph and do global search 

			//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_tt;
			//store the explored sequences as a tree in a deque
			for (size_t i = 0; i < xfers.size(); i++) {
				std::vector<std::set<Op, OpCompare>> affected_ranges_tt;
				Op op_tmp;
				std::set<Op, OpCompare> near_ops_tmp;
				//std::set<Op, OpCompare> empty_range;
				//candidates[0].childrenNode.push_back(std::vector<size_t>());
				//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
				//xfers[i]->run_reuse(0, bestGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tt, add_counter, -1, -1, 0);
				
				//RECORD THE START OF SUBST
				auto time_s = std::chrono::system_clock::now();

				xfers[i]->run_reuse(0, bestGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tt, op_tmp, near_ops_tmp, add_counter, -1, 0, gentime);
			
				//RECORD THE END OF SUBST
				auto time_e = std::chrono::system_clock::now();
				auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
				double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
				run_function_time += searchtime_in;
			}
			this_level_best_change = get_best_change(candidates);
		}
		else
			sample_best_local = false;

		level_num++;
	}


	while (!candidates.empty()) {
		Graph *subGraph = candidates.front().graphPtr;
		candidates.pop_front();
		//delete subGraph;
		delete_graph(original_graph, subGraph);
	}


	//if (print_subst) {
	// printf("        ===== Applied Substitutions =====\n\n");
	// for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
	//  printf("        substitution[%03zu]: \n", i);
	//  Graph::GraphSubst subst = bestGraph->subst_history[i];
	//  for (size_t j = 0; j < subst.srcOps.size(); j++) {
	//	  printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
	//  }
	//  for (size_t j = 0; j < subst.dstOps.size(); j++) {
	//	  printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
	//  }
	// }
	//}

	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	if (print_subst) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: cost change: %.4lf\n", i, bestGraph->subst_history[i].cost_change);
		Graph::GraphSubst subst = bestGraph->subst_history[i];
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        add_counter: %zu\n", add_counter);

	printf("        reuse_counter: %zu\n", reuse_counter);

	//////////////////CODE print cost range of sampled nodes in each level
	/*for (int cost_range_i = 0; cost_range_i < bestGraph->subst_history.size(); cost_range_i++) {
	printf("the cost range in level %d     good nodes [%.4lf, %.4lf]     bad nodes [%.4lf, %.4lf]\n", cost_range_i, lowerbound[cost_range_i * 2], upperbound[cost_range_i * 2], lowerbound[cost_range_i * 2 + 1], upperbound[cost_range_i * 2 + 1]);
	}*/
	/////////////////END CODE

	////////////////
	return bestGraph;
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH NEW REUSE----------------------//
Graph* Graph::optimize_sampletrick_newreuse(float alpha, int budget, bool print_subst, int sample_size)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);



	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//calculate the distance from a node in the source graph of a rule to another node in the source graph
	int max_range = -1;
	for (size_t i = 0; i < xfers.size(); i++) {
		//printf("xferrrrrrrrrrrr: %zu\n", i);
		int tmp = get_distance_matrix(xfers[i]);
		if (tmp > max_range)
			max_range = tmp;
	}

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<SearchTreeNodesample> candidates;
	std::deque<SearchTreeNode> candidates;

	std::vector<std::set<size_t>> hashmaps; //assign a hashmap for each level

											//////////////////////////

											//SearchTreeNodesample rootNode(this);
	SearchTreeNode rootNode(this, -1);
	//candidates.push_back(this);
	//store the sample quota
	rootNode.sample_quota = (size_t)sample_size;
	candidates.push_back(rootNode);
	//////////////////////////

	//hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;


	Graph *bestGraph = this;
	//float bestCost = total_cost();
	//below variable records the bestCost which is updated every time a node is finding its child nodes
	float bestCost_for_sample = total_cost();
	//float best_step_cost_reduc = 0; // the largest cost reduction by one step currently

	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();


	size_t add_counter = 1; //how many searchTreeNodes in candidates have been pushed
	size_t del_counter = 0; //how many searchTreeNodes in candidates have been deleted
	size_t reuse_counter = 0;

	int hop_num = 1; //only find steps which are 1 hop away

	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");


	//assign a hashmap set for level 1
	std::set<size_t> hashmap_level_1;
	hashmaps.push_back(hashmap_level_1);


	//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_tmp;
	//store the explored sequences as a tree in a deque
	for (size_t i = 0; i < xfers.size(); i++) {
		std::vector<std::set<Op, OpCompare>> affected_ranges_tmp;
		Op op_tmp;
		std::set<Op, OpCompare> near_ops_tmp;
		//std::set<Op, OpCompare> empty_range;
		//candidates[0].childrenNode.push_back(std::vector<size_t>());
		//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
		//xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, add_counter, -1, -1, 0);
		
		//RECORD THE START OF SUBST
		auto time_s = std::chrono::system_clock::now();

		xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, op_tmp, near_ops_tmp, add_counter, -1, 0, gentime);
	
		//RECORD THE END OF SUBST
		auto time_e = std::chrono::system_clock::now();
		auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
		double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		run_function_time += searchtime_in;
	}

	// we need to search dependent substitutions following the child substitutions which increases the cost immediately now



	bool sample_best_local = false; //if true, sample the best step and the steps near it


								   ///////////////////////////////CODE for re-allocating sample quota
								   //size_t level_start = 0; //the original index of the start node in a level
	size_t level_end = 0; //the original index of the end node in a level
						  ///////////////////////////////END CODE for re-allocating sample quota

						  ///////////////////////CODE record the cost range of the sampled nodes in each level
						  /*std::vector<float> upperbound;
						  std::vector<float> lowerbound;*/
						  ///////////////////////END CODE

						  //we first pop the first node (with original graph)
	candidates.pop_front();
	del_counter++;

	int level_num = 1;

	float last_level_best_change = -1;
	float this_level_best_change = get_best_change(candidates);

	//rewrite the loop structure for this algo: sample trick local version
	while (!candidates.empty())
	{
		if (level_num >= 2 * budget)
			break;

		// sample the nodes of this level
		//printf("\n=============NEW LEVEL %d=============\n", level_num);

		// if the last step is to increase the cost, we need to directly find the next step which depends on it
		size_t level_size = candidates.size(); //the total number of nodes in this level
		for (size_t i = 0; i < level_size; i++) {
			size_t tem_pos = i;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			if (subGraph->subst_history.back().cost_change <= 0) {
				continue;
			}

			//print whether this graph is sampled or not
			//printf("\n---------------SELF: %zu    FURTHER SEARCH-----------", del_counter + i);// , candidates.at(tem_pos).sample_quota);
																							   //print subst history infor for each graph, no matter whether it is sampled or nor
			//print_subst_history_n_cost(subGraph);

			//candidates.pop_front();

			//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				continue;
			}


			//find all near ops and affected range of them
			//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			//std::vector<Op> the_near_ops;
			//get_affected_range_near(affected_ranges, subGraph, 0, the_near_ops, max_range);
			//std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			//// try to find the child nodes of this node which are NEAR
			////this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			//for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
			//	//this_node.childrenNode.push_back(std::vector<size_t>());

			//	for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
			//		for (size_t newopid = 0; newopid < the_near_ops.size(); newopid++) {
			//			//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);
			//			xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, -1, tem_pos);
			//		}
			//	}
			//}


			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				std::vector<std::set<Op, OpCompare>> affected_ranges_tmp;
				Op op_tmp;
				std::set<Op, OpCompare> near_ops_tmp;
				//std::set<Op, OpCompare> empty_range;
				//candidates[0].childrenNode.push_back(std::vector<size_t>());
				//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
				//xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, add_counter, -1, -1, 0);
				
				//RECORD THE START OF SUBST
				auto time_s = std::chrono::system_clock::now();

				xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges_tmp, op_tmp, near_ops_tmp, add_counter, -1, tem_pos, gentime);
			
				//RECORD THE END OF SUBST
				auto time_e = std::chrono::system_clock::now();
				auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
				double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
				run_function_time += searchtime_in;
			}
		}


		if (sample_best_local)
			do_sampling_best_local(level_end, del_counter, sample_size, candidates, hop_num);
		else
			do_sampling_two_increases(level_end, del_counter, sample_size, candidates);


		//update level_start and level_end
		//level_start = level_end + 1;
		level_end = candidates.size() - 1 + del_counter;

		//assign a hashmap set for the next level to be searched
		std::set<size_t> hashmap_level_next;
		hashmaps.push_back(hashmap_level_next);

		//update last_level_best_change
		last_level_best_change = this_level_best_change;

		//then we search child nodes of all nodes in this level
		level_size = candidates.size(); //the total number of nodes in this level
		for (size_t i = 0; i < level_size; i++) {
			size_t tem_pos = 0;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			if (print_subst) {
				//print whether this graph is sampled or not
				printf("\n---------------SELF: %zu    SAMPLE QUOTA: %zu-----------", del_counter, candidates.at(tem_pos).sample_quota);
				//print subst history infor for each graph, no matter whether it is sampled or nor
				print_subst_history_n_cost(subGraph);
			}
			

			//update bestCost_for_sample
			if (subGraph->total_cost() < bestCost_for_sample) {
				//delete bestGraph;
				delete_graph(original_graph, bestGraph);

				bestCost_for_sample = subGraph->total_cost();
				bestGraph = subGraph;

			}

			if (candidates.at(tem_pos).sample_quota == 0) {
				//there is no quota for this child node, i.e., it is not sampled

				if (subGraph != bestGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}

				candidates.pop_front();
				del_counter++;

				continue;
			}
			//////////////////////////////////////////////////////////////////////

			//candidates.pop_front();

			//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				if (subGraph != bestGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}

				candidates.pop_front();
				del_counter++;

				continue;
			}


			//find all near ops and affected range of them
			//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			//std::vector<Op> the_near_ops;
			//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
			//std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			//// try to find the child nodes of this node which are NEAR
			////this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			//for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
			//	//this_node.childrenNode.push_back(std::vector<size_t>());

			//	for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
			//		for (size_t newopid = 0; newopid < the_near_ops.size(); newopid++) {
			//			//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);
			//			xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, tem_pos);
			//		}
			//	}
			//}


			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				std::vector<std::set<Op, OpCompare>> affected_ranges_tmp;
				Op op_tmp;
				std::set<Op, OpCompare> near_ops_tmp;
				//std::set<Op, OpCompare> empty_range;
				//candidates[0].childrenNode.push_back(std::vector<size_t>());
				//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
				//xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, add_counter, -1, -1, 0);
				
				//RECORD THE START OF SUBST
				auto time_s = std::chrono::system_clock::now();
				
				xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges_tmp, op_tmp, near_ops_tmp, add_counter, -1, tem_pos, gentime);
			
				//RECORD THE END OF SUBST
				auto time_e = std::chrono::system_clock::now();
				auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
				double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
				run_function_time += searchtime_in;
			}


			//xfers[0]->reuse_all_steps(candidates, last_to_search, add_counter, reuse_counter);

			//this_node.last_child = candidates.size() + del_counter; //update the child node range of this node


			//delete the subGraph if it is not the best
			if (subGraph != bestGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}

			candidates.pop_front();
			del_counter++;
		}

		//update this_level_best_change
		this_level_best_change = get_best_change(candidates);

		

		level_num++;
	}


	while (!candidates.empty()) {
		Graph *subGraph = candidates.front().graphPtr;
		candidates.pop_front();
		//delete subGraph;
		delete_graph(original_graph, subGraph);
	}


	//if (print_subst) {
	// printf("        ===== Applied Substitutions =====\n\n");
	// for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
	//  printf("        substitution[%03zu]: \n", i);
	//  Graph::GraphSubst subst = bestGraph->subst_history[i];
	//  for (size_t j = 0; j < subst.srcOps.size(); j++) {
	//	  printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
	//  }
	//  for (size_t j = 0; j < subst.dstOps.size(); j++) {
	//	  printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
	//  }
	// }
	//}

	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	//if (print_subst) {
	if (true) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: cost change: %.4lf\n", i, bestGraph->subst_history[i].cost_change);
		Graph::GraphSubst subst = bestGraph->subst_history[i];
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        add_counter: %zu\n", add_counter);

	printf("        reuse_counter: %zu\n", reuse_counter);

	//////////////////CODE print cost range of sampled nodes in each level
	/*for (int cost_range_i = 0; cost_range_i < bestGraph->subst_history.size(); cost_range_i++) {
	printf("the cost range in level %d     good nodes [%.4lf, %.4lf]     bad nodes [%.4lf, %.4lf]\n", cost_range_i, lowerbound[cost_range_i * 2], upperbound[cost_range_i * 2], lowerbound[cost_range_i * 2 + 1], upperbound[cost_range_i * 2 + 1]);
	}*/
	/////////////////END CODE

	////////////////
	return bestGraph;
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH NEW REUSE which search the steps dependent on a cost-increasing step directly----------------------//
//fatherNode: we need do further search for all child nodes of the father node
//void further_search_after_up_substs(SearchTreeNode& fatherNode, std::deque<SearchTreeNode>& candidates, float alpha, int budget, const std::vector<GraphXfer*>& xfers, std::vector<std::set<size_t>>& hashmaps, float& bestCost_for_sample, size_t& add_counter, size_t del_counter, int maxNumOps, ) {
//
//}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH NEW REUSE which search the steps dependent on a cost-increasing step directly----------------------//
Graph* Graph::optimize_sampletrick_newreuse_2step(float alpha, int budget, bool print_subst, int sample_size)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);



	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//calculate the distance from a node in the source graph of a rule to another node in the source graph
	int max_range = -1;
	for (size_t i = 0; i < xfers.size(); i++) {
		//printf("xferrrrrrrrrrrr: %zu\n", i);
		int tmp = get_distance_matrix(xfers[i]);
		if (tmp > max_range)
			max_range = tmp;
	}

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<SearchTreeNodesample> candidates;
	std::deque<SearchTreeNode> candidates;

	std::vector<std::set<size_t>> hashmaps; //assign a hashmap for each level

											//////////////////////////

											//SearchTreeNodesample rootNode(this);
	SearchTreeNode rootNode(this, -1);
	//candidates.push_back(this);
	//store the sample quota
	rootNode.sample_quota = (size_t)sample_size;
	candidates.push_back(rootNode);
	//////////////////////////

	//hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;


	Graph *bestGraph = this;
	//float bestCost = total_cost();
	//below variable records the bestCost which is updated every time a node is finding its child nodes
	float bestCost_for_sample = total_cost();
	//float best_step_cost_reduc = 0; // the largest cost reduction by one step currently

	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();


	size_t add_counter = 1; //how many searchTreeNodes in candidates have been pushed
	size_t del_counter = 0; //how many searchTreeNodes in candidates have been deleted
	size_t reuse_counter = 0;

	//int hop_num = 1; //only find steps which are 1 hop away

	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

						//ofstream timer_fs;
						//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

						//											 //record old cost
						//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");


	//assign a hashmap set for level 1
	std::set<size_t> hashmap_level_1;
	hashmaps.push_back(hashmap_level_1);


	//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_tmp;
	//store the explored sequences as a tree in a deque
	candidates.front().first_child = candidates.size() + del_counter;
	for (size_t i = 0; i < xfers.size(); i++) {
		std::vector<std::set<Op, OpCompare>> affected_ranges_tmp;
		Op op_tmp;
		std::set<Op, OpCompare> near_ops_tmp;
		//std::set<Op, OpCompare> empty_range;
		//candidates[0].childrenNode.push_back(std::vector<size_t>());
		//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
		//xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, add_counter, -1, -1, 0);

		//RECORD THE START OF SUBST
		auto time_s = std::chrono::system_clock::now();

		xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, op_tmp, near_ops_tmp, add_counter, -1, 0, gentime);

		//RECORD THE END OF SUBST
		auto time_e = std::chrono::system_clock::now();
		auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
		double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		run_function_time += searchtime_in;
	}
	candidates.front().last_child = candidates.size() + del_counter;

	/////////////////////FURTHER SEARCH BEGIN---------------------------------------------------------------------------------------------------
	//for the child nodes which increases the cost, we need to do further search
	SearchTreeNode& fatherNode_furthersearch = candidates.front();
	for (size_t i = fatherNode_furthersearch.first_child; i < fatherNode_furthersearch.last_child; i++) {
		size_t tem_pos = i - del_counter;
		SearchTreeNode& this_node = candidates.at(tem_pos);
		Graph* subGraph = this_node.graphPtr;

		if (subGraph->subst_history.back().cost_change <= 0) {
			continue;
		}

		//we only do further search for child nodes which increase the cost

		//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
		if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
		{
			continue;
		}


		//find all near ops and affected range of them
		std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
		get_affected_range(subGraph, max_range, affected_ranges);
		//std::vector<Op> the_near_ops;
		//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
		const std::vector<Op>& the_near_ops = subGraph->subst_history.back().dstOps;
		std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


		////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

		// try to find the child nodes of this node which are NEAR
		this_node.further_search = true;
		this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
		for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
			//this_node.childrenNode.push_back(std::vector<size_t>());

			for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
				for (size_t newopid = 0; newopid < subGraph->subst_history.back().dstOps.size(); newopid++) {
					//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);

					//RECORD THE START OF SUBST
					auto time_s = std::chrono::system_clock::now();

					xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, i, gentime);

					//RECORD THE END OF SUBST
					auto time_e = std::chrono::system_clock::now();
					auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
					double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
					run_function_time += searchtime_in;
				}
			}
		}

		this_node.last_child = candidates.size() + del_counter; //update the child node range of this node
	}

	//then we need to update the child range of the fatherNode_furthersearch
	fatherNode_furthersearch.last_child = candidates.size() + del_counter;
	/////////////////////FURTHER SEARCH END-----------------------------------------------------------------------------------------------------

	//bool sample_best_local = false; //if true, sample the best step and the steps near it


	///////////////////////////////CODE for re-allocating sample quota
	size_t level_start = 0; //the original index of the start node in a level
	size_t level_end = 0; //the original index of the end node in a level
	///////////////////////////////END CODE for re-allocating sample quota

	while (!candidates.empty())
	{

		SearchTreeNode& fatherNode = candidates.front();

		if (del_counter == level_start) {
			// sample the nodes of this level
			//printf("\n=============NEW LEVEL %zu=============\n", candidates.back().graphPtr->subst_history.size());
			
			do_sampling_two_increases_nocheck_2hash(level_end, del_counter, sample_size, candidates);

			//update level_start and level_end
			level_start = level_end + 1;
			level_end = candidates.size() - 1 + del_counter;

			//assign a hashmap set for the next level to be searched
			std::set<size_t> hashmap_level_next;
			hashmaps.push_back(hashmap_level_next);
		}

		//update last_level_best_change
		//last_level_best_change = this_level_best_change;

		if (fatherNode.further_search) {
			//because the child nodes of this nodes are regarded as the child nodes of its father, and they have searched their child nodes
			candidates.pop_front();
			del_counter++;
			continue;
		}


		//then we search child nodes of all nodes in this level
		//level_size = candidates.size(); //the total number of nodes in this level
		//for (size_t i = 0; i < level_size; i++) {
		for (size_t i = fatherNode.first_child; i < fatherNode.last_child; i++) {
			size_t tem_pos = i - del_counter;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			if (print_subst) {
				//print whether this graph is sampled or not
				printf("\n---------------SELF: %zu  FATHER: %zu   SAMPLE QUOTA: %zu-----------", i, del_counter, candidates.at(tem_pos).sample_quota);
				//print subst history infor for each graph, no matter whether it is sampled or nor
				print_subst_history_n_cost(subGraph);
			}


			//update bestCost_for_sample
			if (subGraph->total_cost() < bestCost_for_sample) {
				//delete bestGraph;
				//delete_graph(original_graph, bestGraph);

				bestCost_for_sample = subGraph->total_cost();
				bestGraph = subGraph;

			}

			if (candidates.at(tem_pos).sample_quota == 0) {
				//there is no quota for this child node, i.e., it is not sampled

				//if (subGraph != bestGraph) {
				//	//delete subGraph;
				//	delete_graph(original_graph, subGraph);
				//}

				/*candidates.pop_front();
				del_counter++;*/

				continue;
			}
			//////////////////////////////////////////////////////////////////////

			//candidates.pop_front();

			//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				//if (subGraph != bestGraph) {
				//	//delete subGraph;
				//	delete_graph(original_graph, subGraph);
				//}

				//candidates.pop_front();
				//del_counter++;

				continue;
			}


			//find all near ops and affected range of them
			std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			get_affected_range(subGraph, max_range, affected_ranges);
			//std::vector<Op> the_near_ops;
			//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
			const std::vector<Op>& the_near_ops = subGraph->subst_history.back().dstOps;
			std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			// try to find the child nodes of this node which are NEAR
			this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < subGraph->subst_history.back().dstOps.size(); newopid++) {
						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);

						//RECORD THE START OF SUBST
						auto time_s = std::chrono::system_clock::now();

						xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, i, gentime);

						//RECORD THE END OF SUBST
						auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;
					}
				}
			}

			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();

			xfers[0]->reuse_all_steps(candidates, tem_pos, add_counter, reuse_counter, del_counter, gentime);

			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;

			this_node.last_child = candidates.size() + del_counter; //update the child node range of this node

			/////////////////////FURTHER SEARCH BEGIN---------------------------------------------------------------------------------------------------
			//for the child nodes which increases the cost, we need to do further search
			SearchTreeNode& fatherNode_furthersearch_inner = this_node;
			for (size_t i_in = fatherNode_furthersearch_inner.first_child; i_in < fatherNode_furthersearch_inner.last_child; i_in++) {
				size_t tem_pos_in = i_in - del_counter;
				SearchTreeNode& this_node_in = candidates.at(tem_pos_in);
				Graph* subGraph_in = this_node_in.graphPtr;

				if (subGraph_in->subst_history.back().cost_change <= 0) {
					continue;
				}

				//we only do further search for child nodes which increase the cost

				//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
				if ((budget > 0) && (subGraph_in->subst_history.size() >= (size_t)budget))
				{
					continue;
				}


				//find all near ops and affected range of them
				std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_in;
				get_affected_range(subGraph_in, max_range, affected_ranges_in);
				//std::vector<Op> the_near_ops;
				//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
				const std::vector<Op>& the_near_ops_in = subGraph_in->subst_history.back().dstOps;
				std::set<Op, OpCompare> all_selected_ops_in(the_near_ops_in.begin(), the_near_ops_in.end());


				////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

				// try to find the child nodes of this node which are NEAR
				this_node_in.further_search = true;
				this_node_in.first_child = candidates.size() + del_counter; //update the child node range of this node
				for (size_t xfer_i_in = 0; xfer_i_in < xfers.size(); xfer_i_in++) {
					//this_node.childrenNode.push_back(std::vector<size_t>());

					for (size_t depthid_in = 0; depthid_in < xfers[xfer_i_in]->srcOps.size(); depthid_in++) {
						for (size_t newopid_in = 0; newopid_in < subGraph_in->subst_history.back().dstOps.size(); newopid_in++) {
							//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);

							//RECORD THE START OF SUBST
							auto time_s_in = std::chrono::system_clock::now();

							xfers[xfer_i_in]->run_reuse(0, subGraph_in, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i_in, affected_ranges_in[newopid_in], the_near_ops_in[newopid_in], all_selected_ops_in, add_counter, depthid_in, i_in, gentime);

							//RECORD THE END OF SUBST
							auto time_e_in = std::chrono::system_clock::now();
							auto duration_in_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e_in - time_s_in);
							double searchtime_in_in = double(duration_in_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
							run_function_time += searchtime_in_in;
						}
					}
				}

				this_node_in.last_child = candidates.size() + del_counter; //update the child node range of this node
			}

			//then we need to update the child range of the fatherNode_furthersearch_inner
			fatherNode_furthersearch_inner.last_child = candidates.size() + del_counter;

			//printf("child first: %zu   child last: %zu\n", fatherNode_furthersearch_inner.first_child, fatherNode_furthersearch_inner.last_child);
			/////////////////////FURTHER SEARCH END-----------------------------------------------------------------------------------------------------




																	//delete the subGraph if it is not the best
																	//if (subGraph != bestGraph) {
																	//	//delete subGraph;
																	//	delete_graph(original_graph, subGraph);
																	//}

																	//candidates.pop_front();
																	//del_counter++;
		}

		// we can release all child graphs and pop out father_node now
		for (size_t i = fatherNode.first_child; i < fatherNode.last_child; i++) {
			size_t tem_pos = i - del_counter;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;
			//delete the subGraph if it is not the best
			if (subGraph != bestGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
		}

		candidates.pop_front();
		del_counter++;
		//update this_level_best_change
		//this_level_best_change = get_best_change(candidates);



		//level_num++;
	}


	while (!candidates.empty()) {
		Graph *subGraph = candidates.front().graphPtr;
		candidates.pop_front();
		//delete subGraph;
		delete_graph(original_graph, subGraph);
	}


	//if (print_subst) {
	// printf("        ===== Applied Substitutions =====\n\n");
	// for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
	//  printf("        substitution[%03zu]: \n", i);
	//  Graph::GraphSubst subst = bestGraph->subst_history[i];
	//  for (size_t j = 0; j < subst.srcOps.size(); j++) {
	//	  printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
	//  }
	//  for (size_t j = 0; j < subst.dstOps.size(); j++) {
	//	  printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
	//  }
	// }
	//}

	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	//if (print_subst) {
	if (true) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: cost change: %.4lf\n", i, bestGraph->subst_history[i].cost_change);
		Graph::GraphSubst subst = bestGraph->subst_history[i];
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        add_counter: %zu\n", add_counter);

	printf("        reuse_counter: %zu\n", reuse_counter);


	////////////////
	return bestGraph;
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH NEW REUSE which search the steps dependent on a sampled cost-increasing step directly----------------------//
void update_candidates(std::deque<SearchTreeNode>& candidates, size_t curr_pos, size_t& del_counter, Graph* original_graph, Graph *bestGraph) {
	//we pop out all nodes before the father of the curr_pos node, and delete the graph of all brother nodes of the curr_pos node
	SearchTreeNode& currNode = candidates.at(curr_pos - del_counter);
	SearchTreeNode& fatherNode = candidates.at(currNode.father - del_counter);
	
	for (size_t i = fatherNode.first_child; i < fatherNode.last_child; i++)
	{
		size_t bro_pos = i - del_counter;
		SearchTreeNode& bro_node = candidates.at(bro_pos);
		Graph* subGraph = bro_node.graphPtr;

		if (subGraph == NULL)
			break;

		//delete the subGraph if it is not the best
		if (subGraph != bestGraph) {
			//delete subGraph;
			delete_graph(original_graph, subGraph);
			bro_node.graphPtr = NULL;
		}
	}

	//if this father has child but quota = 0; i.e., this father is further searched, then we leave it there
	if (fatherNode.sample_quota != 0){
		while (del_counter <= (size_t)(currNode.father))
		{
			candidates.pop_front();
			del_counter++;
		}
	}

}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH NEW REUSE which search the steps dependent on a sampled cost-increasing step directly----------------------//
Graph* Graph::optimize_sampletrick_newreuse_2samplestep(float alpha, int budget, bool print_subst, int sample_size, bool do_weight_process)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);



	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//calculate the distance from a node in the source graph of a rule to another node in the source graph
	int max_range = -1;
	for (size_t i = 0; i < xfers.size(); i++) {
		//printf("xferrrrrrrrrrrr: %zu\n", i);
		int tmp = get_distance_matrix(xfers[i]);
		if (tmp > max_range)
			max_range = tmp;
	}

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<SearchTreeNodesample> candidates;
	std::deque<SearchTreeNode> candidates;

	std::vector<std::set<size_t>> hashmaps; //assign a hashmap for each level

											//////////////////////////

											//SearchTreeNodesample rootNode(this);
	SearchTreeNode rootNode(this, -1);
	//candidates.push_back(this);
	//store the sample quota
	rootNode.sample_quota = (size_t)sample_size;
	candidates.push_back(rootNode);
	//////////////////////////

	//hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;


	Graph *bestGraph = this;
	//float bestCost = total_cost();
	//below variable records the bestCost which is updated every time a node is finding its child nodes
	float bestCost_for_sample = total_cost();
	//float best_step_cost_reduc = 0; // the largest cost reduction by one step currently

	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();


	size_t add_counter = 1; //how many searchTreeNodes in candidates have been pushed
	size_t del_counter = 0; //how many searchTreeNodes in candidates have been deleted
	size_t reuse_counter = 0;

	//int hop_num = 1; //only find steps which are 1 hop away

	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

						//ofstream timer_fs;
						//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

						//											 //record old cost
						//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");


	//assign a hashmap set for level 1
	std::set<size_t> hashmap_level_1;
	hashmaps.push_back(hashmap_level_1);


	//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_tmp;
	//store the explored sequences as a tree in a deque
	candidates.front().first_child = candidates.size() + del_counter;
	for (size_t i = 0; i < xfers.size(); i++) {
		std::vector<std::set<Op, OpCompare>> affected_ranges_tmp;
		Op op_tmp;
		std::set<Op, OpCompare> near_ops_tmp;
		//std::set<Op, OpCompare> empty_range;
		//candidates[0].childrenNode.push_back(std::vector<size_t>());
		//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
		//xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, add_counter, -1, -1, 0);

		//RECORD THE START OF SUBST
		auto time_s = std::chrono::system_clock::now();

		xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, op_tmp, near_ops_tmp, add_counter, -1, 0, gentime);

		//RECORD THE END OF SUBST
		auto time_e = std::chrono::system_clock::now();
		auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
		double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		run_function_time += searchtime_in;
	}
	candidates.front().last_child = candidates.size() + del_counter;


	///////////////////////////////CODE for re-allocating sample quota
	size_t level_start = 0; //the original index of the start node in a level
	size_t level_end = 0; //the original index of the end node in a level
						  ///////////////////////////////END CODE for re-allocating sample quota

	while (!candidates.empty())
	{
		
		// sample the nodes of this level
		//printf("\n=============NEW LEVEL %zu=============\n", candidates.back().graphPtr->subst_history.size());
		//printf("\n=============NEW LEVEL=============\n");
		
		//before we do sampling, we need first get the potential of cost-increasing nodes
		for (size_t i = level_end + 1; i < candidates.size(); i++) {
			size_t tem_pos = i - del_counter;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			if (subGraph->subst_history.back().cost_change <= 0) {
				//if this node does not need further search
				continue;
			}


			//WE NEED TO GET THE POTENTIAL OF IT
			this_node.potential = subGraph->total_cost();


			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				continue;
			}


			//find all near ops and affected range of them
			std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			get_affected_range(subGraph, max_range, affected_ranges);
			//std::vector<Op> the_near_ops;
			//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
			const std::vector<Op>& the_near_ops = subGraph->subst_history.back().dstOps;
			std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			// try to find the child nodes of this node which are NEAR
			//this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < subGraph->subst_history.back().dstOps.size(); newopid++) {
						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);

						//RECORD THE START OF SUBST
						//auto time_s = std::chrono::system_clock::now();

						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, i, gentime);
						xfers[xfer_i]->get_potential(0, subGraph, candidates, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, i, this_node.potential);


						//RECORD THE END OF SUBST
						/*auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;*/
					}
				}
			}

			//RECORD THE START OF SUBST
			//auto time_s = std::chrono::system_clock::now();

			//xfers[0]->reuse_all_steps(candidates, tem_pos, add_counter, reuse_counter, del_counter, gentime);

			//RECORD THE END OF SUBST
			/*auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;*/

			//this_node.last_child = candidates.size() + del_counter; //update the child node range of this node
		}

		do_sampling_2independent_hash_nocheck(level_end, del_counter, sample_size, candidates);

		//update level_start and level_end
		level_start = level_end + 1;
		level_end = candidates.size() - 1 + del_counter;

		//iterate over all nodes in the new level, find child nodes for sampled nodes which need further search; delete graph which is not sampled
		for (size_t i = level_start; i <= level_end; i++) {
			size_t tem_pos = i - del_counter;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			//if (print_subst) {
			//	//print whether this graph is sampled or not
			//	printf("\n---------------SELF: %zu  FATHER: %zu   SAMPLE QUOTA: %zu-----------", i, del_counter, candidates.at(tem_pos).sample_quota);
			//	//print subst history infor for each graph, no matter whether it is sampled or nor
			//	print_subst_history_n_cost(subGraph);
			//}


			//update bestCost_for_sample
			if (subGraph->total_cost() < bestCost_for_sample) {
				//delete bestGraph;
				//delete_graph(original_graph, bestGraph);

				bestCost_for_sample = subGraph->total_cost();
				bestGraph = subGraph;

			}

			if (this_node.sample_quota == 0) {
				//there is no quota for this child node, i.e., it is not sampled

				if (candidates.at(this_node.father - del_counter).has_sampled_child == false) {
					//delete the subGraph if it is not the best
					if (subGraph != bestGraph) {
						//delete subGraph;
						delete_graph(original_graph, subGraph);
						this_node.graphPtr = NULL;
					}
				}
				continue;
			}
			//////////////////////////////////////////////////////////////////////

			if (subGraph->subst_history.back().cost_change <= 0) {
				//if this node does not need further search
				continue;
			}


			//now we are sure this node need further search, WE PRETEND THIS NODE IS NOT SEARCHED
			this_node.sample_quota = 0;

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				continue;
			}


			//find all near ops and affected range of them
			std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			get_affected_range(subGraph, max_range, affected_ranges);
			//std::vector<Op> the_near_ops;
			//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
			const std::vector<Op>& the_near_ops = subGraph->subst_history.back().dstOps;
			std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			// try to find the child nodes of this node which are NEAR
			this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < subGraph->subst_history.back().dstOps.size(); newopid++) {
						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);

						//RECORD THE START OF SUBST
						auto time_s = std::chrono::system_clock::now();

						xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, i, gentime);

						//RECORD THE END OF SUBST
						auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;
					}
				}
			}

			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();

			xfers[0]->reuse_all_steps(candidates, tem_pos, add_counter, reuse_counter, del_counter, gentime);

			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;

			this_node.last_child = candidates.size() + del_counter; //update the child node range of this node
		}


		//we need to sample the further searched nodes
		do_sampling_down_dependent(level_end, del_counter, sample_size, candidates);

		//update level_end
		level_end = candidates.size() - 1 + del_counter;

		//assign a hashmap set for the next level to be searched
		std::set<size_t> hashmap_level_next;
		hashmaps.push_back(hashmap_level_next);
		
		//if there is nothing in the new level break
		if (level_start > level_end)
			break;

		//update last_level_best_change
		//last_level_best_change = this_level_best_change;

		//WE ITERATE NODES IN THE NEW LEVEL (INCLUDING NODES FIRST UP THEN DOWN) TO SEARCH CHILD NODES
		for (size_t i = level_start; i <= level_end; i++) {
			size_t tem_pos = i - del_counter;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			if (subGraph != NULL) {
				if (print_subst) {
					//print whether this graph is sampled or not
					printf("\n---------------SELF: %zu  FATHER: %d   SAMPLE QUOTA: %zu-----------", i, this_node.father, candidates.at(tem_pos).sample_quota);
					//print subst history infor for each graph, no matter whether it is sampled or nor
					print_subst_history_n_cost(subGraph);
				}


				//update bestCost_for_sample
				if (subGraph->total_cost() < bestCost_for_sample) {
					//delete bestGraph;
					//delete_graph(original_graph, bestGraph);

					bestCost_for_sample = subGraph->total_cost();
					bestGraph = subGraph;

				}
			}
			

			if (candidates.at(tem_pos).sample_quota == 0) {
				//there is no quota for this child node, i.e., it is not sampled

				//if all bro nodes have been searched
				if (i == (candidates.at(this_node.father - del_counter).last_child - 1))
					update_candidates(candidates, i, del_counter, original_graph, bestGraph);

				continue;
			}
			
			//////////////////////////////////////////////////////////////////////


			
			//candidates.pop_front();

			//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				//if all bro nodes have been searched
				if (i == (candidates.at(this_node.father - del_counter).last_child - 1))
					update_candidates(candidates, i, del_counter, original_graph, bestGraph);

				continue;
			}


			//find all near ops and affected range of them
			std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			get_affected_range(subGraph, max_range, affected_ranges);
			//std::vector<Op> the_near_ops;
			//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
			const std::vector<Op>& the_near_ops = subGraph->subst_history.back().dstOps;
			std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			// try to find the child nodes of this node which are NEAR
			this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < subGraph->subst_history.back().dstOps.size(); newopid++) {
						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);

						//RECORD THE START OF SUBST
						auto time_s = std::chrono::system_clock::now();

						xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, i, gentime);

						//RECORD THE END OF SUBST
						auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;
					}
				}
			}

			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();

			xfers[0]->reuse_all_steps(candidates, tem_pos, add_counter, reuse_counter, del_counter, gentime);

			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;

			this_node.last_child = candidates.size() + del_counter; //update the child node range of this node

			//if all bro nodes have been searched
			if (i == (candidates.at(this_node.father - del_counter).last_child - 1))
				update_candidates(candidates, i, del_counter, original_graph, bestGraph);

		}

	}


	while (!candidates.empty()) {
		Graph *subGraph = candidates.front().graphPtr;
		candidates.pop_front();
		//delete subGraph;
		//delete_graph(original_graph, subGraph);
	}



	if (do_weight_process)
		bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	//if (print_subst) {
	if (true) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: cost change: %.4lf\n", i, bestGraph->subst_history[i].cost_change);
		Graph::GraphSubst subst = bestGraph->subst_history[i];
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        add_counter: %zu\n", add_counter);

	printf("        reuse_counter: %zu\n", reuse_counter);


	////////////////
	return bestGraph;
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH TRUE NEW REUSE ----------------------//
//which_sample: which sample function to use
Graph* Graph::optimize_sampletrick_truenewreuse(int which_sample, float alpha, int budget, bool print_subst, int sample_size)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";

	//////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	//////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);



	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//calculate the distance from a node in the source graph of a rule to another node in the source graph
	int max_range = -1;
	for (size_t i = 0; i < xfers.size(); i++) {
		//printf("xferrrrrrrrrrrr: %zu\n", i);
		int tmp = get_distance_matrix(xfers[i]);
		if (tmp > max_range)
			max_range = tmp;
	}

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<SearchTreeNodesample> candidates;
	std::deque<SearchTreeNode> candidates;

	std::vector<std::set<size_t>> hashmaps; //assign a hashmap for each level

											//////////////////////////

											//SearchTreeNodesample rootNode(this);
	SearchTreeNode rootNode(this, -1);
	//candidates.push_back(this);
	//store the sample quota
	rootNode.sample_quota = (size_t)sample_size;
	candidates.push_back(rootNode);
	//////////////////////////

	//hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;


	Graph *bestGraph = this;
	//float bestCost = total_cost();
	//below variable records the bestCost which is updated every time a node is finding its child nodes
	float bestCost_for_sample = total_cost();
	//float best_step_cost_reduc = 0; // the largest cost reduction by one step currently

	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();


	size_t add_counter = 1; //how many searchTreeNodes in candidates have been pushed
	size_t del_counter = 0; //how many searchTreeNodes in candidates have been deleted
	size_t reuse_counter = 0;

	//int hop_num = 1; //only find steps which are 1 hop away

	int maxNumOps = inEdges.size();

	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");


	//assign a hashmap set for level 1
	std::set<size_t> hashmap_level_1;
	hashmaps.push_back(hashmap_level_1);


	//std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges_tmp;
	//store the explored sequences as a tree in a deque
	candidates.front().first_child = candidates.size() + del_counter;
	for (size_t i = 0; i < xfers.size(); i++) {
		std::vector<std::set<Op, OpCompare>> affected_ranges_tmp;
		Op op_tmp;
		std::set<Op, OpCompare> near_ops_tmp;
		//std::set<Op, OpCompare> empty_range;
		//candidates[0].childrenNode.push_back(std::vector<size_t>());
		//xfers[i]->run_sampletrick(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, empty_range, 0, add_counter, del_counter, reuse_counter, best_step_cost_reduc, false);
		//xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, add_counter, -1, -1, 0);
		
		//RECORD THE START OF SUBST
		auto time_s = std::chrono::system_clock::now();
		
		xfers[i]->run_reuse(0, this, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, i, affected_ranges_tmp, op_tmp, near_ops_tmp, add_counter, -1, 0, gentime);
	
		//RECORD THE END OF SUBST
		auto time_e = std::chrono::system_clock::now();
		auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
		double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		run_function_time += searchtime_in;
	}
	candidates.front().last_child = candidates.size() + del_counter;
	//bool sample_best_local = false; //if true, sample the best step and the steps near it


									///////////////////////////////CODE for re-allocating sample quota
	size_t level_start = 0; //the original index of the start node in a level
	size_t level_end = 0; //the original index of the end node in a level
						  ///////////////////////////////END CODE for re-allocating sample quota


	//int level_num = 1;

	/*float last_level_best_change = -1;
	float this_level_best_change = get_best_change(candidates);*/

	//rewrite the loop structure for this algo: sample trick local version
	while (!candidates.empty())
	{
		/*if (level_num >= 2 * budget)
			break;*/
		
		SearchTreeNode& fatherNode = candidates.front();

		if (del_counter == level_start) {
			// sample the nodes of this level
			//printf("\n=============NEW LEVEL %zu=============\n", candidates.back().graphPtr->subst_history.size());
			if (which_sample == 0)
				do_sampling(level_end, del_counter, sample_size, candidates);
			else if (which_sample == 1)
				do_sampling_nohash(level_end, del_counter, sample_size, candidates);
			else if (which_sample == 2)
				do_sampling_2independent_hash(level_end, del_counter, sample_size, candidates);
			else 
				do_sampling_2dependent_hash(level_end, del_counter, sample_size, candidates);


			//update level_start and level_end
			level_start = level_end + 1;
			level_end = candidates.size() - 1 + del_counter;

			//assign a hashmap set for the next level to be searched
			std::set<size_t> hashmap_level_next;
			hashmaps.push_back(hashmap_level_next);
		}

		//update last_level_best_change
		//last_level_best_change = this_level_best_change;

		//then we search child nodes of all nodes in this level
		//level_size = candidates.size(); //the total number of nodes in this level
		//for (size_t i = 0; i < level_size; i++) {
		for (size_t i = fatherNode.first_child; i < fatherNode.last_child; i++){
			size_t tem_pos = i - del_counter;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;

			if (print_subst) {
				//print whether this graph is sampled or not
				printf("\n---------------SELF: %zu  FATHER: %zu   SAMPLE QUOTA: %zu-----------", i, del_counter, candidates.at(tem_pos).sample_quota);
				//print subst history infor for each graph, no matter whether it is sampled or nor
				print_subst_history_n_cost(subGraph);
			}
			

			//update bestCost_for_sample
			if (subGraph->total_cost() < bestCost_for_sample) {
				//delete bestGraph;
				//delete_graph(original_graph, bestGraph);

				bestCost_for_sample = subGraph->total_cost();
				bestGraph = subGraph;

			}

			if (candidates.at(tem_pos).sample_quota == 0) {
				//there is no quota for this child node, i.e., it is not sampled

				//if (subGraph != bestGraph) {
				//	//delete subGraph;
				//	delete_graph(original_graph, subGraph);
				//}

				/*candidates.pop_front();
				del_counter++;*/

				continue;
			}
			//////////////////////////////////////////////////////////////////////

			//candidates.pop_front();

			//printf("        the current idx of subGraph being checked now: %zu\n", tem_pos);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				//if (subGraph != bestGraph) {
				//	//delete subGraph;
				//	delete_graph(original_graph, subGraph);
				//}

				//candidates.pop_front();
				//del_counter++;

				continue;
			}


			//find all near ops and affected range of them
			std::vector<std::vector<std::set<Op, OpCompare>>> affected_ranges;
			get_affected_range(subGraph, max_range, affected_ranges);
			//std::vector<Op> the_near_ops;
			//get_affected_range_near(affected_ranges, subGraph, hop_num, the_near_ops, max_range);
			const std::vector<Op>& the_near_ops = subGraph->subst_history.back().dstOps;
			std::set<Op, OpCompare> all_selected_ops(the_near_ops.begin(), the_near_ops.end());


			////SearchTreeNodesample& curr_node = candidates.at(tem_pos);

			// try to find the child nodes of this node which are NEAR
			this_node.first_child = candidates.size() + del_counter; //update the child node range of this node
			for (size_t xfer_i = 0; xfer_i < xfers.size(); xfer_i++) {
				//this_node.childrenNode.push_back(std::vector<size_t>());

				for (size_t depthid = 0; depthid < xfers[xfer_i]->srcOps.size(); depthid++) {
					for (size_t newopid = 0; newopid < subGraph->subst_history.back().dstOps.size(); newopid++) {
						//xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, xfer_i, affected_ranges, add_counter, depthid, newopid, last_to_search);
						
						//RECORD THE START OF SUBST
						auto time_s = std::chrono::system_clock::now();
						
						xfers[xfer_i]->run_reuse(0, subGraph, candidates, hashmaps.back(), bestCost_for_sample * alpha, 2 * maxNumOps, xfer_i, affected_ranges[newopid], the_near_ops[newopid], all_selected_ops, add_counter, depthid, i, gentime);
					
						//RECORD THE END OF SUBST
						auto time_e = std::chrono::system_clock::now();
						auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
						double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						run_function_time += searchtime_in;
					}
				}
			}

			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();

			xfers[0]->reuse_all_steps(candidates, tem_pos, add_counter, reuse_counter, del_counter, gentime);

			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;

			this_node.last_child = candidates.size() + del_counter; //update the child node range of this node


			//delete the subGraph if it is not the best
			//if (subGraph != bestGraph) {
			//	//delete subGraph;
			//	delete_graph(original_graph, subGraph);
			//}

			//candidates.pop_front();
			//del_counter++;
		}

		// we can release all child graphs and pop out father_node now
		for (size_t i = fatherNode.first_child; i < fatherNode.last_child; i++) {
			size_t tem_pos = i - del_counter;
			SearchTreeNode& this_node = candidates.at(tem_pos);
			Graph* subGraph = this_node.graphPtr;
			//delete the subGraph if it is not the best
			if (subGraph != bestGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
		}

		candidates.pop_front();
		del_counter++;
		//update this_level_best_change
		//this_level_best_change = get_best_change(candidates);



		//level_num++;
	}


	while (!candidates.empty()) {
		Graph *subGraph = candidates.front().graphPtr;
		candidates.pop_front();
		//delete subGraph;
		delete_graph(original_graph, subGraph);
	}


	//if (print_subst) {
	// printf("        ===== Applied Substitutions =====\n\n");
	// for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
	//  printf("        substitution[%03zu]: \n", i);
	//  Graph::GraphSubst subst = bestGraph->subst_history[i];
	//  for (size_t j = 0; j < subst.srcOps.size(); j++) {
	//	  printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
	//  }
	//  for (size_t j = 0; j < subst.dstOps.size(); j++) {
	//	  printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
	//  }
	// }
	//}

	bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	//if (print_subst) {
	if (true) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: cost change: %.4lf\n", i, bestGraph->subst_history[i].cost_change);
		Graph::GraphSubst subst = bestGraph->subst_history[i];
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        add_counter: %zu\n", add_counter);

	printf("        reuse_counter: %zu\n", reuse_counter);

	//////////////////CODE print cost range of sampled nodes in each level
	/*for (int cost_range_i = 0; cost_range_i < bestGraph->subst_history.size(); cost_range_i++) {
	printf("the cost range in level %d     good nodes [%.4lf, %.4lf]     bad nodes [%.4lf, %.4lf]\n", cost_range_i, lowerbound[cost_range_i * 2], upperbound[cost_range_i * 2], lowerbound[cost_range_i * 2 + 1], upperbound[cost_range_i * 2 + 1]);
	}*/
	/////////////////END CODE

	////////////////
	return bestGraph;
}






//---------------------THIS FUNCTION ONLY FOR SYSMLPARTITION----------------------//
//calculate the vertex weight, then return the vector weight and adjacent matrix to python interface
std::vector<std::vector<int>> Graph::get_vertex_weight()
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";


	///////////////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	///////////////////////////////////////



	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);


	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<Graph*> candidates;


	//std::set<size_t> hashmap;
	//candidates.push(this);
	//hashmap.insert(hash());
	//Graph *bestGraph = this;
	//float bestCost = total_cost();
	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	//print_costs();

	//store the capacity/weight of every vertex
	std::map<Op, int, OpCompare> vertex_weight;

	//int counter = 0;
	//int maxNumOps = inEdges.size();
	//long long start_time = microsecond_timer();
	//record time
	//auto start_time = std::chrono::system_clock::now();

	//ofstream timer_fs;
	////timer_fs.open("timer.txt");
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//printf("\n        ===== Start Cost-Based Backtracking Search =====\n");

	/*Graph *subGraph = candidates.top();
	candidates.pop();*/

	for (size_t i = 0; i < xfers.size(); i++) {
		//xfers[i]->run(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, i, vertex_weight);
		xfers[i]->collect_vertex_weight(0, this, vertex_weight, i);
	}

	//prepare the information to be returned
	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
	int count = 0;
	std::map<Op, int, OpCompare> op2index;
	for (it = inEdges.begin(); it != inEdges.end(); it++) {
		//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
		operators_for_access.push_back(it->first); //store op in inEdges in order
		op2index[it->first] = count++;
	}
	std::vector<std::vector<int>> results;
	//store weight
	//std::vector<int> weights_to_return(operators_for_access.size(), 0);
	std::vector<int> weights_to_return(count, 0);
	results.push_back(weights_to_return);
	std::map<Op, int, OpCompare>::const_iterator vertexw_it;
	for (vertexw_it = vertex_weight.begin(); vertexw_it != vertex_weight.end(); vertexw_it++) {
		int index = op2index[vertexw_it->first];
		results[0][index] = vertexw_it->second;
	}

	//int source_index = -1;
	//int sink_index = -1;
	
	//store adjacent matrix
	for (it = inEdges.begin(); it != inEdges.end(); it++) {
		std::set<Edge, EdgeCompare> list = it->second;
		std::set<Edge, EdgeCompare>::const_iterator it2;
		for (it2 = list.begin(); it2 != list.end(); it2++) {
			Edge e = *it2;
			std::vector<int> edge_to_return;
			if (e.srcOp.ptr != NULL) {
				edge_to_return.push_back(op2index[e.srcOp]);
				edge_to_return.push_back(op2index[e.dstOp]);
				results.push_back(edge_to_return);
			}
			else {
				assert(inEdges.find(e.srcOp) == inEdges.end()); //this is the original op for input/weight
				//source_index = op2index[e.dstOp];
			}
		}
		/*if (outEdges.find(it->first) == outEdges.end())
			sink_index = op2index[it->first];*/
	}

	/*assert((source_index >= 0) && (sink_index >= 0));
	std::vector<int> source_and_sink;
	source_and_sink.push_back(source_index);
	source_and_sink.push_back(sink_index);
	results.push_back(source_and_sink);*/

	//bestGraph = bestGraph->preprocess_weights();
	//printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	/*auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	timer_fs << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	timer_fs.close();*/

	return results;
}


//---------------------THIS FUNCTION ONLY FOR SYSMLPARTITION----------------------//
// this function does optimization with graph partitions
// partitions store the index of operators (obtained in get_vertex_weight function, the same as in op2index) in each partition
Graph* Graph::optimize_partition(float alpha, int budget, bool print_subst, int eraly_stop_num, std::vector<std::vector<std::vector<int>>> partitions, bool do_weight_process)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			//xfers.push_back(GraphXfer::create_separate_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));

	//add two subst rules
	/*xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_convs_concat_split(model, AC_MODE_RELU));*/

	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";


	///////////////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	///////////////////////////////////////


	//delete one rule which is redundant
	//xfers.erase(xfers.begin() + 152);


	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	//std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<Graph*> candidates;


	std::set<size_t> hashmap;
	//candidates.push(this);
	hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;

	Graph *bestGraph = this;
	float bestCost = total_cost();
	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();

	int counter = 0;
	int maxNumOps = inEdges.size();
	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	////timer_fs.open("timer.txt");
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");

	// store the partition infor in each op in the initial graph
	for (size_t par_i = 0; par_i < partitions.size(); par_i++) {
		//std::set<Op, OpCompare> ops_in_par; // store ops in this partition
		std::vector<std::vector<int>>& alledges = partitions[par_i];
		for (size_t e_i = 0; e_i < alledges.size(); e_i++) {
			int s_index = alledges[e_i][0];
			int t_index = alledges[e_i][1];
			Op& s_op = operators_for_access[s_index];
			Op& t_op = operators_for_access[t_index];
			const std::set<Edge, EdgeCompare>& in_list = inEdges[t_op];
			std::set<Edge, EdgeCompare>::const_iterator in_it;
			bool findedge = false;
			for (in_it = in_list.begin(); in_it != in_list.end(); in_it++) {
				if (in_it->srcOp == s_op) {
					par_infor[*in_it] = par_i;
					findedge = true;
				}
			}
			assert(findedge);
		}
		/*for (size_t ele_i = 0; ele_i < partitions[par_i].size(); ele_i++) {
			int op_index = partitions[par_i][ele_i];
			par_infor[operators_for_access[op_index]] = par_i;
		}*/
	}

	//do optimization for each partition
	for (size_t par_i = 0; par_i < partitions.size(); par_i++) {		
		//int counter = 0; 
		std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
		candidates.push(bestGraph); //the only graph in candidates is the current best graph

		while (!candidates.empty()) {
			Graph *subGraph = candidates.top();
			candidates.pop();
			/*Graph *subGraph = candidates.front();
			candidates.pop_front();*/

			if (subGraph->total_cost() < bestCost) {
				//delete bestGraph;
				delete_graph(original_graph, bestGraph);

				bestCost = subGraph->total_cost();
				bestGraph = subGraph;
			}


			//every graph in candidates will be counted
			//if (counter % 1 == 0) {
			//	printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu length: %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size(), subGraph->subst_history.size());
			//	//timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
			//}

			if ((eraly_stop_num > 0) && (counter >= eraly_stop_num)) {
				//it is time of early stop
				//still need to delete this graph if it is not the best one
				if (bestGraph != subGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}
				break;
			}

			//set early stop threshold using time checking directly
			auto check_time = std::chrono::system_clock::now();
			auto check_duration = std::chrono::duration_cast<std::chrono::microseconds>(check_time - start_time);
			double check_searchtime = double(check_duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			if (check_searchtime >= double(60 * 60)){
				//it is time of early stop
				//still need to delete this graph if it is not the best one
				if (bestGraph != subGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}
				break;
			}

			counter++;
			//printf("\nATTENTION!!!!        This is the %d-th node to be searched\n", counter);

			//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
			if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
			{
				//still need to delete this graph if it is not the best one
				if (bestGraph != subGraph) {
					//delete subGraph;
					delete_graph(original_graph, subGraph);
				}
				continue;
			}
			//////////////////////////////


			//if (counter % 1 == 0) {
			//  printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
			//  //timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
			//}
			//counter ++;


			for (size_t i = 0; i < xfers.size(); i++) {
				//for (size_t j = 0; j < xfers[i]->srcOps.size(); j++) {
				//  printf("srcOps[%zu]: type(%d)\n", j, xfers[i]->srcOps[j]->type);
				//}
				//for (size_t j = 0; j < xfers[i]->dstOps.size(); j++) {
				//  printf("dstOps[%zu]: type(%d)\n", j, xfers[i]->dstOps[j]->type);
				//}
				
				//RECORD THE START OF SUBST
				auto time_s = std::chrono::system_clock::now();
				
				xfers[i]->run_partition(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, i, par_i, gentime);
			
				//RECORD THE END OF SUBST
				auto time_e = std::chrono::system_clock::now();
				auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
				double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
				run_function_time += searchtime_in;
			}
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
		}
		
		//we need to delete the remaining graphs if we stop early
		while (!candidates.empty()) {
			Graph *subGraph = candidates.top();
			candidates.pop();
			//delete subGraph;
			delete_graph(original_graph, subGraph);
		}
		
		//now we complete the search in one partition
	}
	
	// now we complete the independent search in each partition. Next, do optimization after stitching all partitions, in fact just do local search 
	//int counter = 0;
	std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	candidates.push(bestGraph); //the only graph in candidates is the current best graph
	while (!candidates.empty()) {
		Graph *subGraph = candidates.top();
		candidates.pop();
		/*Graph *subGraph = candidates.front();
		candidates.pop_front();*/

		if (subGraph->total_cost() < bestCost) {
			//delete bestGraph;
			delete_graph(original_graph, bestGraph);

			bestCost = subGraph->total_cost();
			bestGraph = subGraph;
		}


		//every graph in candidates will be counted
		//if (counter % 1 == 0) {
		//	printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu length: %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size(), subGraph->subst_history.size());
		//	//timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
		//}

		if ((eraly_stop_num > 0) && (counter >= eraly_stop_num)) {
			//it is time of early stop
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			break;
		}

		//set early stop threshold using time checking directly
		auto check_time = std::chrono::system_clock::now();
		auto check_duration = std::chrono::duration_cast<std::chrono::microseconds>(check_time - start_time);
		double check_searchtime = double(check_duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		if (check_searchtime >= double(60 * 60)) {
			//it is time of early stop
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			break;
		}

		counter++;
		//printf("\nATTENTION!!!!        This is the %d-th node to be searched\n", counter);


		//////////////////////////////
		//when budget <= 0, we do not have the budget constraint, and the stopping condition is that all candidates are checked
		//if ((budget > 0) && (counter > budget)) {
		//     // TODO: free all remaining candidates when budget exhausted 
		//     break;
		//   }

		//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
		if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
		{
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			continue;
		}
		//////////////////////////////


		//if (counter % 1 == 0) {
		//  printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
		//  //timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
		//}
		//counter ++;


		for (size_t i = 0; i < xfers.size(); i++) {
			//for (size_t j = 0; j < xfers[i]->srcOps.size(); j++) {
			//  printf("srcOps[%zu]: type(%d)\n", j, xfers[i]->srcOps[j]->type);
			//}
			//for (size_t j = 0; j < xfers[i]->dstOps.size(); j++) {
			//  printf("dstOps[%zu]: type(%d)\n", j, xfers[i]->dstOps[j]->type);
			//}
			
			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();
			
			xfers[i]->run_boundary(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, i, gentime);
		
			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;
		}
		if (bestGraph != subGraph) {
			//delete subGraph;
			delete_graph(original_graph, subGraph);
		}
	}


	//we need to delete the remaining graphs if we stop early
	while (!candidates.empty()) {
		Graph *subGraph = candidates.top();
		candidates.pop();
		//delete subGraph;
		delete_graph(original_graph, subGraph);
	}

	if (do_weight_process)
		bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime,  run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	if (print_subst) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
			printf("        substitution[%03zu]: cost change: %.4lf\n", i, bestGraph->subst_history[i].cost_change);
			Graph::GraphSubst subst = bestGraph->subst_history[i];
			for (size_t j = 0; j < subst.srcOps.size(); j++) {
				printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
			}
			for (size_t j = 0; j < subst.dstOps.size(); j++) {
				printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
			}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        counter: %d\n", counter);
	////////////////
	return bestGraph;
}


//---------------------THIS FUNCTION ONLY FOR SYSMLTRICK WITHOUT PARTITION ----------------------//
Graph* Graph::optimize_sysmltrick(float alpha, int budget, bool print_subst, int eraly_stop_num, bool do_weight_process)
{
	std::vector<GraphXfer*> xfers;
	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 2; j++) {
			PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
			xfers.push_back(GraphXfer::create_conv_relu(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_batch(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_mul(model, i, i, pad_mode));
			xfers.push_back(GraphXfer::create_conv_add(model, i, i, pad_mode));
		}
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
	//xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
	//split enlarge_merge_convs into two steps
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_enlarge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_convs(model, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
	xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

	//xfers.push_back(create_avg_pool_conv(model));
	//xfers.push_back(create_two_pools(model));
	//xfers.push_back(create_merge_seperable_convs(model));
	char* taso_path = getenv("TASO_HOME");
	if (taso_path == NULL) {
		fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
			"Please set TASO_HOME to the home directory of TASO source code.\n");
		assert(false);
	}
	std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";


	///////////////////////////////////////
	GraphXfer::load_graph_xfer_from_pb_file(model, xfers, graph_subst_file);
	///////////////////////////////////////


	//xfers.push_back(create_fuse_conv_batch_xfer(model));
	//xfers.push_back(create_fuse_conv_relu_xfer(model));
	//xfers.push_back(create_merge_conv_xfer(model));
	//xfers.push_back(create_exclusive_concat_xfer(model));
	//xfers.push_back(create_enlarge_conv_xfer(model));
	//xfers.push_back(create_resnet_merge_xfer(model));

	std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
	//changing the "candidates" from priority_queue to deque
	//std::deque<Graph*> candidates;


	std::set<size_t> hashmap;
	candidates.push(this);
	hashmap.insert(hash());

	//record the original graph
	Graph* original_graph = this;

	Graph *bestGraph = this;
	float bestCost = total_cost();
	//printf("MetaFlow Cost = %.4lfms\n", bestCost);
	//printf("Input graph: end-to-end execution time =\n"
	//       "%.8lf ms (average of 100 runs)\n", run());
	print_costs();

	int counter = 0;
	int maxNumOps = inEdges.size();
	//long long start_time = microsecond_timer();
	//record time
	auto start_time = std::chrono::system_clock::now();
	double run_function_time = 0; //the time of executing run_... function
	double gentime = 0; //the time of generating new graph and do some preparation

	//ofstream timer_fs;
	////timer_fs.open("timer.txt");
	//timer_fs.open("results.txt", std::ios::app); //add all new contend to the end of the file, and store old cost, new cost, search time, peak memory may be recorded by shell

	//											 //record old cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
	while (!candidates.empty()) {
		Graph *subGraph = candidates.top();
		candidates.pop();
		/*Graph *subGraph = candidates.front();
		candidates.pop_front();*/

		if (print_subst) {
			//print subst history infor for each graph, no matter whether it is sampled or nor
			print_subst_history_n_cost(subGraph);
		}
		
		
		if (subGraph->total_cost() < bestCost) {
			//delete bestGraph;
			delete_graph(original_graph, bestGraph);

			bestCost = subGraph->total_cost();
			bestGraph = subGraph;
		}


		//every graph in candidates will be counted
		//if (counter % 1 == 0) {
		//	printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu length: %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size(), subGraph->subst_history.size());
		//	//timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
		//}

		if ((eraly_stop_num > 0) && (counter >= eraly_stop_num)) {
			//it is time of early stop
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			break;
		}

		//set early stop threshold using time checking directly
		auto check_time = std::chrono::system_clock::now();
		auto check_duration = std::chrono::duration_cast<std::chrono::microseconds>(check_time - start_time);
		double check_searchtime = double(check_duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		if (check_searchtime >= double(60 * 60)) {
			//it is time of early stop
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			break;
		}

		counter++;


		//////////////////////////////
		//when budget <= 0, we do not have the budget constraint, and the stopping condition is that all candidates are checked
		//if ((budget > 0) && (counter > budget)) {
		//     // TODO: free all remaining candidates when budget exhausted 
		//     break;
		//   }

		//change the "budget" meaning from "the max number of computation graphs to be checked" to "the max length of the subst history of a optimized computation graph"
		if ((budget > 0) && (subGraph->subst_history.size() >= (size_t)budget))
		{
			//still need to delete this graph if it is not the best one
			if (bestGraph != subGraph) {
				//delete subGraph;
				delete_graph(original_graph, subGraph);
			}
			continue;
		}
		//////////////////////////////


		//if (counter % 1 == 0) {
		//  printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
		//  //timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
		//}
		//counter ++;


		for (size_t i = 0; i < xfers.size(); i++) {
			//for (size_t j = 0; j < xfers[i]->srcOps.size(); j++) {
			//  printf("srcOps[%zu]: type(%d)\n", j, xfers[i]->srcOps[j]->type);
			//}
			//for (size_t j = 0; j < xfers[i]->dstOps.size(); j++) {
			//  printf("dstOps[%zu]: type(%d)\n", j, xfers[i]->dstOps[j]->type);
			//}
			
			//RECORD THE START OF SUBST
			auto time_s = std::chrono::system_clock::now();

			xfers[i]->run_sysmltrick(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps, i, gentime);
		
			//RECORD THE END OF SUBST
			auto time_e = std::chrono::system_clock::now();
			auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);
			double searchtime_in = double(duration_in.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			run_function_time += searchtime_in;
		}
		if (bestGraph != subGraph) {
			//delete subGraph;
			delete_graph(original_graph, subGraph);
		}
	}

	//we need to delete the remaining graphs if we stop early
	while (!candidates.empty()) {
		Graph *subGraph = candidates.top();
		candidates.pop();
		//delete subGraph;
		delete_graph(original_graph, subGraph);
	}

	if (do_weight_process)
		bestGraph = bestGraph->preprocess_weights();
	printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");

	//record new cost
	//timer_fs << bestGraph->total_cost() << std::endl;

	//record time
	auto end_time = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	//timer_fs << "result[search_time] = " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << std::endl;
	//timer_fs.close();

	//RECORD RESULTS USING STANDARD FUNCTION
	record_result(original_graph->total_cost(), bestGraph->total_cost(), searchtime, run_function_time - gentime, gentime);

	//printf("bestCost = %.4lf\n", bestGraph->total_cost());
	//printf("Optimized graph: end-to-end execution time =\n");
	//printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
	bestGraph->print_costs();
	//if (print_subst) {
	if (true) {
		/*printf("        ===== Applied Substitutions =====\n\n");
		for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
			printf("        substitution[%03zu]: \n", i);
			Graph::GraphSubst subst = bestGraph->subst_history[i];
			for (size_t j = 0; j < subst.srcOps.size(); j++) {
				printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
			}
			for (size_t j = 0; j < subst.dstOps.size(); j++) {
				printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
			}
		}*/
		print_subst_history_n_cost(bestGraph);
	}

	////////////////
	//print counter
	printf("        counter: %d\n", counter);
	////////////////
	return bestGraph;
}



Graph* Graph::preprocess_weights(void)
{
  Graph* newGraph = new Graph();
  newGraph->subst_history = subst_history;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  // Step 1: clone the input graph
  for (opIt = inEdges.begin(); opIt != inEdges.end(); opIt++)
  {
    const std::set<Edge, EdgeCompare>& list = opIt->second;
    std::set<Edge, EdgeCompare>::const_iterator it;
    for (it = list.begin(); it != list.end(); it++)
      newGraph->add_edge(it->srcOp, it->dstOp, it->srcIdx, it->dstIdx);
  }
  // Step 2: iteratively process the weights
  while (true) {
    bool change = false;
    for (opIt = newGraph->inEdges.begin(); opIt != newGraph->inEdges.end(); opIt++) {
      if (opIt->first.ptr->type == OP_INPUT || opIt->first.ptr->type == OP_WEIGHT)
        continue;
      bool allWeights = true;
      const std::set<Edge, EdgeCompare>& list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (it->srcOp.ptr->type != OP_WEIGHT) {
          allWeights = false;
          break;
        }
      if (allWeights) {
        // Preprocess weights
        // Currently assume the op has single output
        Op op = opIt->first;
        //assert(op.ptr->numOutputs == 1);
        // map and execute the operator to get the output weights
        for (it = list.begin(); it != list.end(); it++) {
          assert(it->srcOp.ptr->outputs[it->srcIdx].data_ptr != NULL);
          assert(op.ptr->inputs[it->dstIdx].has_same_shape_stride_split(
              it->srcOp.ptr->outputs[it->srcIdx]));
          op.ptr->inputs[it->dstIdx].data_ptr =
              it->srcOp.ptr->outputs[it->srcIdx].data_ptr;
        }
        op.ptr->map();
        op.ptr->forward(true/*block*/);
        TensorHandle tensor = newGraph->new_weight(op.ptr->outputs[0]);
        newGraph->replace_node(op, tensor->op);
        op.ptr->unmap();
        newGraph->remove_node(op);
        change = true;
        break;
      }
    }
    // Stop if we didn't make any change
    if (!change)
      break;
  }
  // Remove isolated nodes
  std::map<Op, int, OpCompare> todos;
  std::vector<Op> weightList;
  std::set<Op, OpCompare> weightOps;
  for (opIt = newGraph->inEdges.begin(); opIt != newGraph->inEdges.end(); opIt++) {
    int cnt = 0;
    const std::set<Edge, EdgeCompare>& inList = opIt->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid != GUID_WEIGHT) cnt ++;
    }
    todos[opIt->first] = cnt;
    if (cnt == 0)
      weightList.push_back(opIt->first);
  }
  size_t i = 0;
  while (i < weightList.size()) {
    Op op = weightList[i++];
    weightOps.insert(op);
    const std::set<Edge, EdgeCompare>& outList = newGraph->outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        weightList.push_back(it2->dstOp);
      }
    }
  }
  while (true) {
    bool change = false;
    for (opIt = newGraph->inEdges.begin(); opIt != newGraph->inEdges.end(); opIt++) {
      Op op = opIt->first;
      if (weightOps.find(op) != weightOps.end() && newGraph->num_out_edges(op) == 0) {
        newGraph->remove_node(op);
        change = true;
        break;
      }
    }
    if (!change)
      break;
  }
  return newGraph;
}

Op Graph::find_op_or_fail(size_t guid)
{
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++)
    if (it->first.guid == guid) {
      return it->first;
    }
  assert(false);
}

int Graph::get_operator_list(Op* ops, size_t maxNumOps)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }

  size_t cnt = 0, i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    if ((op.ptr->type == OP_INPUT) || (op.ptr->type == OP_WEIGHT)) {
    } else {
      ops[cnt++] = op;
    }
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
  assert(opList.size() == inEdges.size());
  return cnt;
}

int Graph::get_input_edges(Edge* ops, size_t guid)
{
  Op op = find_op_or_fail(guid);
  assert(inEdges.find(op) != inEdges.end());
  std::set<Edge, EdgeCompare> inList = inEdges[op];
  size_t cnt = inList.size();
  std::set<Edge, EdgeCompare>::const_iterator it2;
  for (it2 = inList.begin(); it2 != inList.end(); it2 ++) {
    Edge e = *it2;
    ops[it2->dstIdx] = e;
  }
  // We manually delete the second input for pool2d
  if (op.ptr->type == OP_POOL2D_MAX || op.ptr->type == OP_POOL2D_AVG) {
    assert(cnt == 2 || cnt == 1);
    cnt = 1;
  }
  return cnt;
}

OpType Graph::get_operator_type(size_t guid)
{
  Op op = find_op_or_fail(guid);
  return op.ptr->type;
}

int Graph::get_operator_int_attr(size_t guid, PMParameter attr)
{
  Op op = find_op_or_fail(guid);
  int ret;
  assert(op.ptr->get_int_parameter(attr, &ret));
  return ret;
}

int Graph::get_num_outputs(size_t guid)
{
  Op op = find_op_or_fail(guid);
  return op.ptr->numOutputs;
}

int Graph::get_input_dims(size_t guid, int* dims, int idx)
{
  Op op = find_op_or_fail(guid);
  assert(op.ptr->numInputs > idx);
  int ndim = op.ptr->inputs[idx].numDim;
  for (int i = 0; i < ndim; i++)
    dims[i] = op.ptr->inputs[idx].dim[i];
  return ndim;
}

void Graph::get_weight_value(size_t guid, DATATYPE* value)
{
  Op op = find_op_or_fail(guid);
  // Assume weight op has one input and one output
  assert(op.ptr->type == OP_WEIGHT);
  assert(op.ptr->numInputs == 1);
  assert(op.ptr->numOutputs == 1);
  assert(op.ptr->inputs[0].data_ptr != NULL);
  model->copy_memory(value, (DATATYPE*) op.ptr->inputs[0].data_ptr,
      sizeof(DATATYPE) * op.ptr->inputs[0].volume());
}

int Graph::get_output_dims(size_t guid, int* dims, int idx)
{
  Op op = find_op_or_fail(guid);
  assert(op.ptr->numOutputs > idx);
  int ndim = op.ptr->outputs[idx].numDim;
  for (int i = 0; i < ndim; i++)
    dims[i] = op.ptr->outputs[idx].dim[i];
  return ndim;
}

int Graph::get_split_lens(size_t guid, int* lens)
{
  Op op = find_op_or_fail(guid);
  assert(op.ptr->type == OP_SPLIT);
  Split* split = (Split*) op.ptr;
  int numSplits = split->numOutputs;
  for (int i = 0; i < numSplits; i++)
    lens[i] = split->outputs[i].dim[split->axis];
  return numSplits;
}

void Graph::add_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx)
{
  assert(dstOp.guid != OP_WEIGHT);
  if (inEdges.find(dstOp) == inEdges.end()) {
    inEdges[dstOp];
  }
  if (outEdges.find(srcOp) == outEdges.end()) {
    outEdges[srcOp];
  }
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  inEdges[dstOp].insert(e);
  outEdges[srcOp].insert(e);
}

void Graph::remove_edge(Edge e)
{
  assert(outEdges[e.srcOp].find(e) != outEdges[e.srcOp].end());
  assert(inEdges[e.dstOp].find(e) != inEdges[e.dstOp].end());
  assert(outEdges[e.srcOp].erase(e) == 1);
  assert(inEdges[e.dstOp].erase(e) == 1);
}

void Graph::replace_node(Op oldOp, Op newOp)
{
  //if (outEdges.find(newOp) == outEdges.end()) {
  //  outEdges[newOp];
  //}
  const std::set<Edge, EdgeCompare>& outSet = outEdges[oldOp];
  std::set<Edge, EdgeCompare>::const_iterator it;
  std::vector<Edge> outList;
  for (it = outSet.begin(); it != outSet.end(); it++)
    outList.push_back(*it);
  for (size_t i = 0; i < outList.size(); i++) {
    Edge e = outList[i];
    remove_edge(e);
    add_edge(newOp, e.dstOp, e.srcIdx, e.dstIdx);
  }
}

void Graph::remove_node(Op oldOp)
{
  assert(outEdges.find(oldOp) != outEdges.end());
  // Asser that it is safe to remove the node
  assert(outEdges[oldOp].size() == 0);
  const std::set<Edge, EdgeCompare>& inSet = inEdges[oldOp];
  std::set<Edge, EdgeCompare>::const_iterator it;
  std::vector<Edge> inList;
  for (it = inSet.begin(); it != inSet.end(); it++)
    inList.push_back(*it);
  for (size_t i = 0; i < inList.size(); i++)
    remove_edge(inList[i]);
  assert(inEdges[oldOp].size() == 0);
  inEdges.erase(oldOp);
  outEdges.erase(oldOp);
}

// We do this in topological order because it will be easier to parse on
// the other end
void Graph::export_to_file(std::string file_name)
{
  ofstream export_fs;
  export_fs.open(file_name.c_str());
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
    {
      opList.push_back(it->first);
    }
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    export_op(export_fs, op);

    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) opList.push_back(it2->dstOp);
    }
  }
  export_fs.close();
  assert(opList.size() == inEdges.size());
}

/* Exports an operator with the following format:
 * guid
 * type
 * dependencies (comma separated list of other ops)
 * parameters (comma separated and type dependent)
 */
void Graph::export_op(ofstream &file_stream, Op &op)
{
  file_stream << op.guid << std::endl;

  file_stream << op.ptr->type << std::endl;

  std::string deps_string;
  std::set<Edge, EdgeCompare> inList = inEdges[op];
  std::set<Edge, EdgeCompare>::const_iterator it;
  int i = 0;
  for (it = inList.begin(); it != inList.end(); it++) {
    deps_string += std::to_string(it->srcOp.guid);
    deps_string += ':';
    deps_string += std::to_string(it->srcIdx);
    deps_string += ',';
    i++;
  }
  if (deps_string.size() > 0)
  {
    deps_string = deps_string.substr(0, deps_string.size()-1);
  }
  file_stream << deps_string.c_str() << std::endl;

  switch (op.ptr->type) {
    case OP_CONV2D:
    { 
      Conv2D* conv = (Conv2D*) op.ptr;
      Tensor t = conv->inputs[0];
      Tensor w = conv->inputs[1];
      int padH, padW;
      conv->get_padding(&padH, &padW);
      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << t.dim[3] << ','; // 3
      file_stream << w.dim[0] << ','; // 4
      file_stream << w.dim[1] << ','; // 5
      file_stream << w.dim[2] << ','; // 6
      file_stream << w.dim[3] << ','; // 7
      file_stream << conv->strideH << ','; // 8
      file_stream << conv->strideW << ','; // 9
      file_stream << conv->padding << ','; // 10
      file_stream << conv->activation << ','; // 11
      file_stream << padH << ','; // 12
      file_stream << padW; // 13
      break;
    }
    case OP_POOL2D_MAX:
    case OP_POOL2D_AVG:
    {
      Pool2D* pool = (Pool2D*) op.ptr;
      Tensor t = pool->inputs[0];
      int padH, padW;
      pool->get_padding(&padH, &padW);
      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << t.dim[3] << ','; // 3
      file_stream << pool->type << ','; // 4
      file_stream << pool->kernelH << ','; // 5
      file_stream << pool->kernelW << ','; // 6
      file_stream << pool->strideH << ','; // 7
      file_stream << pool->strideW << ','; // 8
      file_stream << pool->padding << ','; // 9
      file_stream << pool->activation << ','; // 10
      file_stream << padH << ','; // 11
      file_stream << padW; // 12
      break;
    }
    case OP_SPLIT:
    {
      Split* split = (Split*) op.ptr;
      file_stream << split->axis << ',';
      for (int i = 0; i < split->numOutputs; i++)
      {
        file_stream << split->sizes[i];
        if (i < split->numOutputs - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    case OP_CONCAT:
    {
      Concat* concat = (Concat*) op.ptr;
      file_stream << concat->axis;
      //TODO: fix below for visualizer
      //Tensor t = concat->inputs[0];
      //file_stream << t.dim[0] << ','; // 0
      //file_stream << t.dim[1] << ','; // 1
      //file_stream << t.dim[2] << ','; // 2
      //file_stream << t.dim[3]; // 3
      break;
    }
    case OP_EW_ADD:
    case OP_EW_MUL:
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
    case OP_BATCHNORM:
    case OP_INPUT:
    case OP_WEIGHT:
    {
      Tensor t = op.ptr->inputs[0];
      for (int i = 0; i < t.numDim; i++)
      {
        file_stream << t.dim[i]; // 0 - N
        if (i < t.numDim - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    case OP_MATMUL: // This doesn't seem to be implemented in run either
    {
      Matmul* matmul = (Matmul*) op.ptr;
      file_stream << matmul->activation << ','; // 0
      file_stream << matmul->outputs[0].numDim; // 1
      break;
    }
    case OP_RESHAPE:
    {
      //Reshape *reshape = (Reshape*) op.ptr;
      Tensor t = op.ptr->outputs[0];
      for (int i = 0; i < t.numDim; i++)
      {
        file_stream << t.dim[i]; // 0 - N
        if (i < t.numDim - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    case OP_TRANSPOSE:
    {
      Transpose *transpose = (Transpose*) op.ptr;
      Tensor t = op.ptr->outputs[0];
      int permIdx = transpose->permIdx;
      int ndim = t.numDim;
      //int permArray[MAX_DIM];
      for (int i = ndim - 1; i >= 0; i--) {
        //permArray[i] = permIdx % ndim;
        permIdx = permIdx / ndim;
      }
      assert(permIdx == 0);
      for (int i = 0; i < ndim; i++) {
        file_stream << t.dim[i];// 0 - N
        if (i < ndim - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    default:
      assert(false);
  }
  file_stream << std::endl;
}

size_t Graph::num_in_edges(Op op)
{
  return inEdges[op].size();
}

size_t Graph::num_out_edges(Op op)
{
  return outEdges[op].size();
}

bool Graph::has_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx)
{
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges[dstOp].find(e) != inEdges[dstOp].end());
}

size_t Graph::hash(void)
{
  size_t total = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    size_t my = 17 * 31 + (size_t)(it->first.ptr);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      my = my * 31 + std::hash<size_t>()((size_t)(e.srcOp.ptr));
      my = my * 31 + std::hash<int>()(e.srcIdx);
      my = my * 31 + std::hash<int>()(e.dstIdx);
    }
    total += my;
  }
  return total;
}

void Graph::print(void)
{
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    if (it->first.guid == 0) continue;
    printf("	guid(%zu) type(%d) runtime(%.4lf): ", it->first.guid, it->first.ptr->type, it->first.ptr->runtime);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      printf(" inEdge(guid(%zu) idx(%d))", e.srcOp.guid, e.srcIdx);
    }
    printf("\n");
  }
}

bool Graph::check_correctness(void)
{
  bool okay = true;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx)) assert(false);
      if (e.srcOp.ptr == NULL) continue;
      Tensor srcTensor = e.srcOp.ptr->outputs[e.srcIdx];
      Tensor dstTensor = e.dstOp.ptr->inputs[e.dstIdx];
      if (srcTensor.numDim != dstTensor.numDim) assert(false);
      for (int i = 0; i < srcTensor.numDim; i++) {
        if (srcTensor.dim[i] != dstTensor.dim[i]) {
          assert(false);
          return false;
        }
        if (srcTensor.stride[i] != dstTensor.stride[i]) {
          //assert(false);
          //return false;
        }
      }
    }
  }
  return okay;
}

float Graph::total_cost(void)
{
  if (totalCost > 0) return totalCost;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  float total = 0.0f;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    if (it->first.ptr != NULL) total += it->first.ptr->runtime;
  }
  totalCost = total;
  return total;
}

bool Graph::has_loop(void)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
  return (opList.size() < inEdges.size());
}

float Graph::run(void)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  std::vector<OpBase*> opBaseList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare> inList = inEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    assert(inList.size() > 0);
    OpBase* opPtr = NULL;
    // Step 1: prepare inputs
    Tensor inputs[MAX_NUM_INPUTS];
    if ((op.ptr->type == OP_INPUT) || (op.ptr->type == OP_WEIGHT)) {
      assert(inList.size() == 1);
      //Edge e = *inList.begin();
      //assert(e.srcOp.ptr == NULL); // NoOp's input must not be any Op
      Tensor t = op.ptr->inputs[0];
      size_t size = sizeof(DATATYPE);
      for (int j = 0; j < t.numDim; j++)
        size *= t.dim[j];
      if (op.ptr->type == OP_INPUT) {
        assert(t.data_ptr == NULL);
        t.data_ptr = (DATATYPE*) model->allocate_memory(size);
      } else {
        assert(t.data_ptr != NULL);
      }
      inputs[0] = t;
    } else {
      for (it2 = inList.begin(); it2 != inList.end(); it2++) {
        size_t idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == it2->srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[it2->dstIdx].data_ptr == NULL); // No duplicated dstIdxes
        inputs[it2->dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }
#ifdef DEADCODE
    // Step 1: prepare inputs
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      Edge e = *it2;
      if (e.srcOp.guid == GUID_INPUT) {
        Tensor t = op.ptr->inputs[e.dstIdx];
        t.ptr = (DATATYPE*) model->allocate_memory(sizeof(DATATYPE) * t.size());
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = t;
      } else if (e.srcOp.guid = GUID_WEIGHT) {
        Tensor t = op.ptr->inputs[e.dstIdx];
        t.ptr = (DATATYPE*) model->allocate_memory(sizeof(DATATYPE) * t.size());
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = t;
      } else {
        size_t idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == e.srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }
#endif
    // Step 2: create Ops
    switch (op.ptr->type) {
      case OP_CONV2D:
      {
        Conv2D* conv = (Conv2D*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Conv2D(model, inputs[0], inputs[1],
                           conv->strideH, conv->strideW,
                           conv->padding, conv->activation);
#ifdef USE_CUDNN
        ((Conv2D*)opPtr)->fwdAlgo = conv->fwdAlgo;
#endif
        break;
      }
      case OP_MATMUL:
      {
        Matmul* matmul = (Matmul*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Matmul(model, inputs[0], inputs[1], matmul->activation);
        break;
      }
      case OP_RESHAPE:
      {
        Reshape* reshape = (Reshape*) op.ptr;
        assert(inList.size() == 1);
        std::vector<int> shape;
        for (int i = 0; i < reshape->outputs[0].numDim; i++)
          shape.push_back(reshape->outputs[0].dim[i]);
        opPtr = new Reshape(model, inputs[0], shape);
        break;
      }
      case OP_TRANSPOSE:
      {
        Transpose* transpose = (Transpose*) op.ptr;
        assert(inList.size() == 1);
        int ndim = inputs[0].numDim, permIdx = transpose->permIdx;
        std::vector<int> permVec;
        int permArray[MAX_DIM];
        for (int i = ndim - 1; i >= 0; i--) {
          permArray[i] = permIdx % ndim;
          permIdx = permIdx / ndim;
        }
        assert(permIdx == 0);
        for (int i = 0; i < ndim; i++)
          for (int j = i + 1; j < ndim; j++)
            assert(permArray[i] != permArray[j]);
        for (int i = 0; i < ndim; i++)
          permVec.push_back(permArray[i]);
        opPtr = new Transpose(model, inputs[0], permVec, transpose->shuffle);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_MUL:
      {
        //Element* element = (Element*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Element(model, op.ptr->type, inputs[0], inputs[1]);
        break;
      }
      case OP_ENLARGE:
      {
        //Enlarge* enlarge = (Enlarge*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Enlarge(model, inputs[0], inputs[1]);
        break;
      }
      case OP_MERGE_GCONV:
      {
        MergeGConv* merge = (MergeGConv*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new MergeGConv(model, inputs[0], merge->count);
        break;
      }
      case OP_POOL2D_MAX:
      case OP_POOL2D_AVG:
      {
        Pool2D* pool = (Pool2D*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Pool2D(model, inputs[0], inputs[1], pool->type,
                           pool->kernelH, pool->kernelW,
                           pool->strideH, pool->strideW,
                           pool->padding, pool->activation);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      {
        Activation* act = (Activation*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Activation(model, inputs[0], act->type, act->inPlace);
        break;
      }
      case OP_BATCHNORM:
      {
        assert(inList.size() == 5);
        opPtr = new BatchNorm(model, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
        break;
      }
      case OP_SPLIT:
      {
        Split* split = (Split*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Split(model, inputs[0], split->axis, split->sizes);
        break;
      }
      case OP_INPUT:
      case OP_WEIGHT:
      case OP_DROPOUT:
      {
        assert(inList.size() == 1);
        opPtr = new NoOp(model, inputs[0], op.ptr->type);
        break;
      }
      case OP_CONCAT:
      {
        Concat* concat = (Concat*) op.ptr;
        opPtr = new Concat(model, concat->axis, inList.size(), inputs, concat->needCopy);
        break;
      }
      default:
        printf("op.type = %d\n", op.ptr->type);
        assert(false);
    }
    // Step 3: map new Op
    opPtr->map();
    opBaseList.push_back(opPtr);
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      //printf("myOp(%zu) dstOp(%zu) dstType(%d) dstTodos(%d)\n",
      //    it2->srcOp.guid, it2->dstOp.guid,
      //    it2->dstOp.ptr->type, todos[it2->dstOp]);
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
#ifdef VERBOSE_PRINTS
  for (int i =0; i < opList.size(); i++) {
    printf("opList[%d]: guid(%zu) type(%d)\n", i, opList[i].guid,
           opList[i].ptr->type);
  }
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    printf("op: guid(%zu) type(%d)\n", it->first.guid, it->first.ptr->type);
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    int cnt = 0;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      printf("    inEdge[%d]: srcOp(%zu) srcIdx(%d) dstOp(%zu) dstIdx(%d)\n", cnt++, it2->srcOp.guid, it2->srcIdx, it2->dstOp.guid, it2->dstIdx);
    }
  }
#endif

  assert(opList.size() == inEdges.size());
  assert(opList.size() == opBaseList.size());

  return model->measure_oplist_runtime(opBaseList);
}

void Graph::print_costs(void)
{
  float exe_time = 0, flops = 0, mem_acc = 0;
  int num_kernels = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++)
    it->first.ptr->collect_costs(exe_time, flops, mem_acc, num_kernels);
  printf("        Cost metrics: exe_time(%.4lf) flops(%.4lf) "
         "memory_access(%.4lf) kernel_launches(%d)\n",
         exe_time, flops / 1024.0 / 1024.0 / 1024.0,
         mem_acc * 4.0 / 1024.0 / 1024.0, num_kernels);
}

