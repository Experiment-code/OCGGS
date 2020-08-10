/* Copyright 2018 Stanford
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

#ifndef _SUBSTITUTION_H_
#define _SUBSTITUTION_H_
#include "taso/ops.h"
#include "rules.pb.h"
#include <queue>
namespace taso {

enum Compare {
  COMPARE_EQ,
  COMPARE_NE,
  COMPARE_LT,
  COMPARE_LE,
  COMPARE_GT,
  COMPARE_GE,
};

struct PMConstraint {
  PMConstraint(Compare comp, PMParameter para, int value);
  Compare comp;
  PMParameter para;
  int value;
};

struct TNConstraint {
  TNConstraint(Compare comp, TNParameter para, DIMParameter dim, int value);
  TNConstraint(Compare comp, TNParameter para1, DIMParameter dim1,
               TNParameter para2, DIMParameter dim2);
  bool singlePara;
  Compare comp;
  TNParameter para1, para2;
  DIMParameter dim1, dim2;
  int value;
};

class OpX;
class GraphXfer;
struct TensorX {
  TensorX(void): op(NULL), idx(0) {}
  TensorX(OpX* _op, int _idx): op(_op), idx(_idx) {}
  Tensor to_tensor(const GraphXfer* xfer) const;
  OpX* op;
  int idx;
};

struct TensorXCompare {
  bool operator()(const TensorX& a, const TensorX& b) const {
    if (a.op != b.op) return a.op < b.op;
    return a.idx < b.idx;
  };
};

class OpX {
public:
  OpX(const OpX& _op);
  OpX(OpType _type, TensorX input0, int numOutputs = 1);
  OpX(OpType _type, TensorX input0, TensorX input1);
  OpX(OpType _type, TensorX input0, TensorX input1, TensorX input2, TensorX input3, TensorX input4);
  OpX(OpType _type, int n, TensorX* ins);
  bool add_pm_constraint(Compare comp, PMParameter para, int value);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, int);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, TNParameter, DIMParameter);
  bool get_pm_constraint(PMParameter para, int& value) const;
public:
  OpType type;
  Op mapOp;
  std::vector<TensorX> inputs, outputs;
  std::vector<PMConstraint> pmConstraints;
  std::vector<TNConstraint> tnConstraints;
};

class DstOp;
class SrcOp {
public:
  SrcOp(OpType _type);
  bool add_constraint(Compare comp, PMParameter para, int value);
  bool match(Op op);
public:
  std::vector<PMConstraint> constraints;
  OpType type;
  Op mapOp;
  DstOp *mapInput, *mapOutput;
};

class DstOp {
public:
  DstOp(OpType _type);
  DstOp(OpType _type, const SrcOp* op);
  DstOp(OpType _type, const SrcOp* op1, const SrcOp* op2);
  virtual Op create_operator(Model* model) = 0;
public:
  OpType type;
  Op mapOp;
  SrcOp *mapInput, *mapOutput;
  SrcOp *srcOps[MAX_NUM_INPUTS];
};

template <typename OpType>
struct SubEdge {
  SubEdge(OpType* _srcOp, OpType* _dstOp, int _srcIdx, int _dstIdx)
  : srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx) {}
  int srcIdx, dstIdx;
  OpType *srcOp, *dstOp;
};

template<typename OpType>
struct SubEdgeCompare {
  bool operator()(const SubEdge<OpType>& a, const SubEdge<OpType>& b) const {
    if (a.srcOp != b.srcOp) return a.srcOp < b.srcOp;
    if (a.dstOp != b.dstOp) return a.dstOp < b.dstOp;
    if (a.srcIdx != b.srcIdx) return a.srcIdx < b.srcIdx;
    if (a.dstIdx != b.dstIdx) return a.dstIdx < b.dstIdx;
    return false;
  };
};

class GraphCompare {
public:
  bool operator() (Graph* lhs, Graph* rhs) {
    return lhs->total_cost() > rhs->total_cost();
  }
};


//--------------------------------THE STRUCTURE ONLY FOR REUSE----------------------------//
//define a helper struct: represent a node in the search tree
struct SearchTreeNode
{
	SearchTreeNode(void);
	SearchTreeNode(Graph* _graphPtr, int fatherid);
	Graph* graphPtr; //pointing to the optimized graph at this stage
					 //std::vector<std::vector<size_t>> childrenNode; //the original index in the candidates of search tree nodes corresponding to the children nodes
					 //GraphXfer meta_data; //store information to use when optimizing graph with explored results 
	int father;
	//int grandpa; //seems we do not need this value
	bool searched; // denote whether the child nodes of this node has been searched or not
	size_t first_child; //the index of the first child node of this node in the deque list, initial value = -1, which is invalid
	size_t last_child; //the index of the last child node of this node in the deque list + 1, initial value = -1, which is invalid
					   //we can only store first and last index of the child nodes because all child nodes are stored sequentially in the candidates deque


	size_t sample_quota; //store the sample quota for this node

	bool further_search; //true: the node is cost-increasing and its child nodes are searched immediately after it before sampling; otherwise, false
	bool has_sampled_child; //true: has sampled child nodes; otherwise, false
	bool reused; //true: this node is generated by reusing
	float potential; //the potential value for cost-up nodes

	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> map_output_infor; //store the output tensor mapping infor, each edge is from the old tensor to the new tensor

																		   //std::map<Op, OpX*, OpCompare> meta_mappedOps;
																		   //std::multimap<int, std::pair<Op, int> > meta_mappedInputs;
																		   //std::map<TensorX, TensorX, TensorXCompare> meta_mappedOutputs;
																		   //std::vector<OpX*> meta_dstOps;
																		   //Graph::GraphSubst subst; // store the last step of each graph optimization sequence this node represents
};



//--------------------------------THE STRUCTURE ONLY FOR SAMPLETRICK----------------------------//
//define a helper struct: represent a node in the search tree
struct SearchTreeNodesample
{
	SearchTreeNodesample(void);
	SearchTreeNodesample(Graph* _graphPtr);
	Graph* graphPtr; //pointing to the optimized graph at this stage
	std::vector<std::vector<size_t>> childrenNode; //the original index in the candidates of search tree nodes corresponding to the children nodes
												   //GraphXfer meta_data; //store information to use when optimizing graph with explored results 
	std::map<Op, OpX*, OpCompare> meta_mappedOps;
	std::multimap<int, std::pair<Op, int> > meta_mappedInputs;
	std::map<TensorX, TensorX, TensorXCompare> meta_mappedOutputs;
	std::vector<OpX*> meta_dstOps;
	//std::vector<size_t> child_sample_quota; //store the sample quota for each child nodes of this search node
	//std::vector<float> child_sample_weight; //store the sample weight for each child node of this search node
	size_t sample_quota; //store the sample quota for this node
	Graph::GraphSubst subst; // store the last step of each graph optimization sequence this node represents
};


//--------------------------------THE STRUCTURE ONLY FOR SAMPLETRICK----------------------------//
//////////////////code for compare two search tree nodes
class SearchTreeNodesampleCompare {
public:
	bool operator() (SearchTreeNodesample* lhs, SearchTreeNodesample* rhs) {
		return lhs->graphPtr->total_cost() < rhs->graphPtr->total_cost();
	}
};
//////////////////end code

//--------------------------------THE STRUCTURE ONLY FOR SAMPLETRICK----------------------------//
//////////////////code for compare two search tree nodes
class SearchTreeNodeCompare {
public:
	bool operator() (SearchTreeNode* lhs, SearchTreeNode* rhs) {
		return lhs->graphPtr->total_cost() < rhs->graphPtr->total_cost();
	}
};
//////////////////end code


//--------------------------------THE STRUCTURE ONLY FOR SAMPLETRICK consider potential----------------------------//
//////////////////code for compare two search tree nodes
class SearchTreeNodepotentialCompare {
public:
	bool operator() (SearchTreeNode* lhs, SearchTreeNode* rhs) {
		return lhs->potential < rhs->potential;
	}
};
//////////////////end code



class GraphXfer {
public:
  GraphXfer(Model* _model);
  static void load_graph_xfer_from_pb_file(Model* model,
                                           std::vector<GraphXfer*>& xfers,
                                           std::string filename);
  TensorX new_tensor(void);
  bool can_match(OpX* srcOp, Op op, Graph* graph);
  void match(OpX* srcOp, Op op, Graph* graph);
  void unmatch(OpX* srcOp, Op op, Graph* graph);
  void create_operator_from_pb(const GraphSubst::Operator& pbOp,
                               std::map<int, TensorX>& mappedInputs,
                               bool isSrcOp = true);
  OpX* create_activation(TensorX input, OpType type, bool isSrcOp = true);
  OpX* create_conv2d(TensorX input, TensorX weight,
                     //int kernelH, int kernelW,
                     int strideH, int strideW,
                     PaddingMode padding,
                     ActiMode activation,
                     bool isSrcOp = true);
  OpX* create_batchnorm(TensorX input, TensorX scale, TensorX bias, TensorX mean, TensorX var);
  OpX* create_element(TensorX input0, TensorX input1,
                      OpType type, bool isSrcOp = true);
  OpX* create_pool2d_avg(TensorX input, TensorX weight,
                         //int kernelH, int kernelW,
                         int strideH, int strideW,
                         PaddingMode padding,
                         ActiMode activation,
                         bool isSrcOp = true);
  OpX* create_matmul(TensorX input, TensorX weight,
                     ActiMode activation, bool isSrcOp = true);
  OpX* create_mul(TensorX x, TensorX y, bool isSrcOp = true);
  OpX* create_transpose(TensorX input, int numDim, int* perm, int shuffle);
  OpX* create_enlarge(TensorX w1, TensorX w2, bool isSrcOp = true);
  OpX* create_enlarge_a(TensorX w1, TensorX w2, bool isSrcOp = true); //the input constraint is different from create_enlarge
  //OpX* create_enlarge_b(TensorX w1, TensorX w2, bool isSrcOp = true); //the input constraint is different from create_enlarge
  OpX* create_merge_gconv(TensorX w, int count, bool isSrcOp = true);
  OpX* create_concat(int axis, int numDim, TensorX in1, TensorX in2, bool isSrcOp = true);
  OpX* create_concat(int axis, int numDim, int n, TensorX* ins, bool isSrcOp = true);
  OpX* create_split(TensorX input, int axis, int n, bool isSrcOp = true);
  void add_src_op(SrcOp* op);
  void add_dst_op(DstOp* op);
  void add_src_edge(SrcOp* src, SrcOp* tgt, int srcIdx = 0, int dstIdx = 0);
  void add_dst_edge(DstOp* src, DstOp* tgt, int srcIdx = 0, int dstIdx = 0);
  bool add_constraint(Compare comp, SrcOp* src, PMParameter srcPara,
                      SrcOp* tgt, PMParameter dstPara);
  bool map_input(SrcOp* src, DstOp* dst);
  bool map_output(SrcOp* src, DstOp* dst);
  bool map_output(TensorX src, TensorX dst);
  /*void run(int depth, Graph* graph,
           std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>&,
           std::set<size_t>&, float threshold, int maxNumOps, int substtype);*/
  
  //--------------------common functions for all algorithms----------------------//
  bool create_dst_ops_update_op_dict(int substtype, Graph* graph);
  bool create_dst_ops_update_op_dict_SYS(int substtype, Graph* graph);
		   
  //--------------------common functions for partial order----------------------//
  bool check(Graph* graph);
  int compStep(Graph::GraphSubst& last);
  int compNode(const Op& firstNode, const Op& secondNode);
  Graph* create_new_graph_order(Graph* graph, int substType);
  //--------------------prune---------------------------//
  void run_prune(int depth, Graph* graph,
	  std::forward_list<Graph*>& candidates,
	  std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType, double& substtime);
  //--------------------enumeration---------------------------//
  void run_enumeration(int depth, Graph* graph,
	  std::forward_list<Graph*>& candidates,
	  std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType, double& substtime);
  //--------------------reuse---------------------------//
  void reuse_all_steps(std::deque<SearchTreeNode>& candidates, int curr_node_id, size_t& add_counter, size_t& reuse_counter, double& substtime);
  void run_reuse(int depth, Graph* graph,
	  std::deque<SearchTreeNode>& candidates,
	  std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType,
	  std::vector<std::vector<std::set<Op, OpCompare>>>& affected_range,
	  size_t& add_counter, int depthid, int newopid, int curr_node_pos, double& substtime);
  void store_infor(SearchTreeNode& search_tree_node);
  //--------------------sampletrick with true new reuse-----------------//
  void reuse_all_steps(std::deque<SearchTreeNode>& candidates, int curr_node_id, size_t& add_counter, size_t& reuse_counter, size_t del_counter, double& substtime);
  //--------------------sampletrick local version and with new reuse and with true new reuse-----------------//
  void run_reuse(int depth, Graph* graph,
	  std::deque<SearchTreeNode>& candidates,
	  std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType,
	  std::vector<std::set<Op, OpCompare>>& affected_range,
	  Op selected_map_op, const std::set<Op, OpCompare>& all_selected_ops,
	  size_t& add_counter, int depthid, int curr_node_pos, double& substtime);
  //--------------------sampletrick get potential-----------------//
  void get_potential(int depth, Graph* graph,
	  std::deque<SearchTreeNode>& candidates,
	  int substType,
	  std::vector<std::set<Op, OpCompare>>& affected_range,
	  Op selected_map_op, const std::set<Op, OpCompare>& all_selected_ops,
	  size_t& add_counter, int depthid, int curr_node_pos, float& potential);

  //--------------------sampletrick-------------------------------//
  void run_sampletrick(int depth, Graph* graph,
	  std::deque<SearchTreeNodesample>& candidates,
	  std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType,
	  std::set<Op, OpCompare>& affected_range,
	  size_t curr_graph_pos, size_t& add_counter, size_t del_counter, size_t& reuse_counter, float& best_step_cost_reduc, bool can_use_reuse, double& substtime);
  void store_infor_sampletrick(SearchTreeNodesample& search_tree_node);
  //--------------------sysml partition---------------------------//
  void run_partition(int depth, Graph* graph,
	  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>&,
	  std::set<size_t>&, float threshold, int maxNumOps, int substtype, int par_i, double& substtime);
  void run_boundary(int depth, Graph* graph,
	  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>&,
	  std::set<size_t>&, float threshold, int maxNumOps, int substtype, double& substtime);
  /*void run(int depth, Graph* graph,
	  std::deque<Graph*>&,
	  std::set<size_t>&, float threshold, int maxNumOps, int substtype);*/
  void collect_vertex_weight(int depth, Graph* graph, std::map<Op, int, OpCompare>& vertex_weight, int substtype);
  //--------------------sysmltrick without partition---------------------------//
  void run_sysmltrick(int depth, Graph* graph,
	  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
	  std::set<size_t>& hashmap, float threshold, int maxNumOps, int substtype, double& substtime);
 
  Graph* create_new_graph_sysmlpartition(Graph* graph);
  
  bool create_new_operator(const OpX* opx, Op& op);

  // built-in substitutions
  static GraphXfer* create_conv_relu(Model* model, int strideH, int strideW, PaddingMode padding);
  // add a reversed rule of create_conv_relu
  //static GraphXfer* create_separate_conv_relu(Model* model, int strideH, int strideW, PaddingMode mode);
  // add a rule which adds concat and split op after two conv ops
  static GraphXfer* create_convs_concat_split(Model* model, ActiMode activation);
  static GraphXfer* create_conv_batch(Model* model, int strideH, int strideW, PaddingMode padding);
  static GraphXfer* create_conv_mul(Model* model, int strideH, int strideW, PaddingMode padding);
  static GraphXfer* create_conv_add(Model* model, int strideH, int strideW, PaddingMode padding);
  static GraphXfer* create_enlarge_merge_convs(Model* model, ActiMode activation);
  //split enlarge_merge_convs into two steps
  static GraphXfer* create_enlarge_convs(Model* model, ActiMode activation);
  static GraphXfer* create_merge_convs(Model* model, ActiMode activation);
  static GraphXfer* create_merge_group_convs(Model* model, int strideH, int strideW, ActiMode activation);
public:
  Model* model;
  int tensorId;
  //std::vector<TwoOpConstraint> constraints;
  //std::map<SrcOp*, std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> > > srcInEdges, srcOutEdges;
  //std::map<DstOp*, std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > > dstInEdges, dstOutEdges;
  std::map<Op, OpX*, OpCompare> mappedOps;
  std::multimap<int, std::pair<Op, int> > mappedInputs;
  std::map<TensorX, TensorX, TensorXCompare> mappedOutputs;
  std::vector<OpX*> srcOps;
  std::vector<OpX*> dstOps;
  //------------TWO MEMBER ONLY FOR REUSE
  Op biggestNode; //which node in the srcOps is the biggest node according to the partial order
  std::vector<std::vector<int>> dis_matrix; //if two nodes are not connected, the distance value is -1
};

} // namespace XFlow
#endif
