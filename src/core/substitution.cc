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

#include "taso/substitution.h"
using namespace taso;


//add a built-in graph substitution rule: adding a concat and a split op after two conv ops
GraphXfer* GraphXfer::create_convs_concat_split(Model* model, ActiMode activation) {
	GraphXfer* subst = new GraphXfer(model);
	TensorX input1 = subst->new_tensor();
	TensorX input2 = subst->new_tensor();
	TensorX w1 = subst->new_tensor();
	TensorX w2 = subst->new_tensor();
	OpX* conv1 = subst->create_conv2d(input1, w1, 1, 1, PD_MODE_SAME, activation);
	OpX* conv2 = subst->create_conv2d(input2, w2, 1, 1, PD_MODE_SAME, activation);
	subst->srcOps.push_back(conv1);
	subst->srcOps.push_back(conv2);

	OpX* conv3 = subst->create_conv2d(input1, w1, 1, 1, PD_MODE_SAME, activation, false/*isSrc*/);
	OpX* conv4 = subst->create_conv2d(input2, w2, 1, 1, PD_MODE_SAME, activation, false/*isSrc*/);

	//OpX* enlarge = subst->create_enlarge(w1, w2, false/*isSrc*/);
	OpX* concat = subst->create_concat(1/*axis*/, 4/*dim*/, conv3->outputs[0],
		conv4->outputs[0], false/*isSrc*/);
	OpX* split = subst->create_split(concat->outputs[0], 1/*axis*/, 2, false/*isSrc*/);

	subst->dstOps.push_back(conv3);
	subst->dstOps.push_back(conv4);
	subst->dstOps.push_back(concat);
	subst->dstOps.push_back(split);

	subst->map_output(conv1->outputs[0], split->outputs[0]);
	subst->map_output(conv2->outputs[0], split->outputs[1]);
	return subst;
}



GraphXfer* create_avg_pool_conv(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX weight = subst->new_tensor();
  OpX* avg_pool = subst->create_pool2d_avg(input, weight, 1, 1,
                                           PD_MODE_SAME,
                                           AC_MODE_NONE);
  OpX* conv = subst->create_conv2d(input, weight, 1, 1,
                                   PD_MODE_SAME,
                                   AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(avg_pool->outputs[0], conv->outputs[0]);
  subst->srcOps.push_back(avg_pool);
  subst->dstOps.push_back(conv);
  return subst;
}

GraphXfer* create_two_pools(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX w1 = subst->new_tensor();
  //TensorX w2 = subst->new_tensor();
  OpX* pool1 = subst->create_pool2d_avg(input, w1, 1, 1,
                                        PD_MODE_SAME,
                                        AC_MODE_NONE);
  //OpX* pool2 = subst->create_pool2d_avg(input, w2, 1, 1,
  //                                      PD_MODE_SAME,
  //                                      AC_MODE_NONE);
  //OpX* add = subst->create_element(pool1->outputs[0], pool2->outputs[0],
  //                                 OP_EW_ADD);
  OpX* pool3 = subst->create_conv2d(input, w1, 1, 1,
                                    PD_MODE_SAME,
                                    AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(pool1->outputs[0], pool3->outputs[0]);
  subst->srcOps.push_back(pool1);
  //subst->srcOps.push_back(pool2);
  //subst->srcOps.push_back(add);
  subst->dstOps.push_back(pool3);
  return subst;
}

GraphXfer* GraphXfer::create_conv_relu(Model* model, int strideH, int strideW, PaddingMode mode)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX weight = subst->new_tensor();
  OpX* conv = subst->create_conv2d(input, weight, strideH, strideW, mode,
                                   AC_MODE_NONE);
  OpX* relu = subst->create_activation(conv->outputs[0], OP_RELU);
  OpX* fuse = subst->create_conv2d(input, weight, strideH, strideW, mode,
                                   AC_MODE_RELU, false/*isSrc*/);
  subst->map_output(relu->outputs[0], fuse->outputs[0]);
  subst->srcOps.push_back(conv);
  subst->srcOps.push_back(relu);
  subst->dstOps.push_back(fuse);
  return subst;
}

//add the reversed rule of create_conv_relu: breaking a conv_relu op into a conv op + a relu op
//GraphXfer* GraphXfer::create_separate_conv_relu(Model* model, int strideH, int strideW, PaddingMode mode)
//{
//	GraphXfer* subst = new GraphXfer(model);
//	TensorX input = subst->new_tensor();
//	TensorX weight = subst->new_tensor();
//	OpX* fuse = subst->create_conv2d(input, weight, strideH, strideW, mode,
//		AC_MODE_RELU);
//
//	OpX* conv = subst->create_conv2d(input, weight, strideH, strideW, mode,
//		AC_MODE_NONE, false/*isSrc*/);
//	OpX* relu = subst->create_activation(conv->outputs[0], OP_RELU, false/*isSrc*/);
//
//	subst->map_output(fuse->outputs[0], relu->outputs[0]);
//	subst->srcOps.push_back(fuse); 
//	subst->dstOps.push_back(conv);
//	subst->dstOps.push_back(relu);
//	return subst;
//}



GraphXfer* GraphXfer::create_conv_batch(Model* model, int strideH, int strideW, PaddingMode mode)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX weight = subst->new_tensor();
  TensorX w[4];
  for (int i = 0; i < 4; i++)
    w[i] = subst->new_tensor();
  OpX* conv = subst->create_conv2d(input, weight, strideH, strideW, mode, AC_MODE_NONE);
  OpX* batch = subst->create_batchnorm(conv->outputs[0], w[0], w[1], w[2], w[3]);
  OpX* fuse = subst->create_conv2d(input, weight, strideH, strideW, mode,
                                   AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(batch->outputs[0], fuse->outputs[0]);
  subst->srcOps.push_back(conv);
  subst->srcOps.push_back(batch);
  subst->dstOps.push_back(fuse);
  return subst;
}

GraphXfer* GraphXfer::create_conv_mul(Model* model, int strideH, int strideW, PaddingMode mode)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX weight = subst->new_tensor();
  TensorX y = subst->new_tensor();
  OpX* conv = subst->create_conv2d(input, weight, strideH, strideW, mode, AC_MODE_NONE);
  OpX* mul = subst->create_element(conv->outputs[0], y, OP_EW_MUL);
  OpX* fuse = subst->create_conv2d(input, weight, strideH, strideW, mode,
                                    AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(mul->outputs[0], fuse->outputs[0]);
  subst->srcOps.push_back(conv);
  subst->srcOps.push_back(mul);
  subst->dstOps.push_back(fuse);
  return subst;
}

GraphXfer* GraphXfer::create_conv_add(Model* model, int strideH, int strideW, PaddingMode mode)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX weight = subst->new_tensor();
  TensorX y = subst->new_tensor();
  OpX* conv = subst->create_conv2d(input, weight, strideH, strideW, mode, AC_MODE_NONE);
  OpX* add = subst->create_element(conv->outputs[0], y, OP_EW_ADD);
  OpX* fuse = subst->create_conv2d(input, weight, strideH, strideW, mode,
                                    AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(add->outputs[0], fuse->outputs[0]);
  subst->srcOps.push_back(conv);
  subst->srcOps.push_back(add);
  subst->dstOps.push_back(fuse);
  return subst;
}

GraphXfer* GraphXfer::create_enlarge_merge_convs(Model* model, ActiMode activation)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX w1 = subst->new_tensor();
  TensorX w2 = subst->new_tensor();
  OpX* conv1 = subst->create_conv2d(input, w1, 1, 1, PD_MODE_SAME, activation);
  OpX* conv2 = subst->create_conv2d(input, w2, 1, 1, PD_MODE_SAME, activation);
  subst->srcOps.push_back(conv1);
  subst->srcOps.push_back(conv2);
  OpX* enlarge = subst->create_enlarge(w1, w2, false/*isSrc*/);
  OpX* concat = subst->create_concat(0/*axis*/, 4/*dim*/, enlarge->outputs[0],
                                     w2, false/*isSrc*/);
  OpX* conv3 = subst->create_conv2d(input, concat->outputs[0], 1, 1,
                                    PD_MODE_SAME, activation, false/*isSrc*/);
  OpX* split = subst->create_split(conv3->outputs[0], 1/*axis*/, 2, false/*isSrc*/);
  subst->dstOps.push_back(enlarge);
  subst->dstOps.push_back(concat);
  subst->dstOps.push_back(conv3);
  subst->dstOps.push_back(split);
  subst->map_output(conv1->outputs[0], split->outputs[0]);
  subst->map_output(conv2->outputs[0], split->outputs[1]);
  return subst;
}



//try to split the enlarge_merge_convs into two steps
//this is the first step, which enlarge a conv kernal
GraphXfer* GraphXfer::create_enlarge_convs(Model* model, ActiMode activation)
{
	GraphXfer* subst = new GraphXfer(model);
	TensorX input = subst->new_tensor();
	TensorX w1 = subst->new_tensor();
	TensorX w2 = subst->new_tensor();
	OpX* conv1 = subst->create_conv2d(input, w1, 1, 1, PD_MODE_SAME, activation);
	OpX* conv2 = subst->create_conv2d(input, w2, 1, 1, PD_MODE_SAME, activation);
	subst->srcOps.push_back(conv1);
	subst->srcOps.push_back(conv2);
	OpX* enlarge = subst->create_enlarge_a(w1, w2, false/*isSrc*/); //the input constraint is different from create_enlarge()
	//OpX* concat = subst->create_concat(0/*axis*/, 4/*dim*/, enlarge->outputs[0],
	//		w2, false/*isSrc*/);
	//OpX* split = subst->create_split(concat->outputs[0], 0/*axis*/, 2, false/*isSrc*/);
	//OpX* conv3 = subst->create_conv2d(input, split->outputs[0], 1, 1, PD_MODE_SAME, activation, false/*isSrc*/); //the convolution op after enlarging
	//OpX* conv4 = subst->create_conv2d(input, split->outputs[1], 1, 1, PD_MODE_SAME, activation, false/*isSrc*/); //the same as the original convolution op: conv2
	OpX* conv3 = subst->create_conv2d(input, enlarge->outputs[0], 1, 1, PD_MODE_SAME, activation, false/*isSrc*/); //the convolution op after enlarging
	OpX* conv4 = subst->create_conv2d(input, w2, 1, 1, PD_MODE_SAME, activation, false/*isSrc*/); //the same as the original convolution op: conv2
	subst->dstOps.push_back(enlarge);
	//subst->dstOps.push_back(concat);
	//subst->dstOps.push_back(split);
	subst->dstOps.push_back(conv3);
	subst->dstOps.push_back(conv4);
	subst->map_output(conv1->outputs[0], conv3->outputs[0]);
	subst->map_output(conv2->outputs[0], conv4->outputs[0]);
	return subst;
}

//this is the second step, which merge two convs with the same kernal size
GraphXfer* GraphXfer::create_merge_convs(Model* model, ActiMode activation)
{
	GraphXfer* subst = new GraphXfer(model);
	TensorX input = subst->new_tensor();
	TensorX w1 = subst->new_tensor();
	TensorX w2 = subst->new_tensor();
	//OpX* split = subst->create_split(w1, 0/*axis*/, 2);
	//OpX* conv1 = subst->create_conv2d(input, split->outputs[0], 1, 1, PD_MODE_SAME, activation);
	//OpX* conv2 = subst->create_conv2d(input, split->outputs[1], 1, 1, PD_MODE_SAME, activation);
	OpX* conv1 = subst->create_conv2d(input, w1, 1, 1, PD_MODE_SAME, activation);
	OpX* conv2 = subst->create_conv2d(input, w2, 1, 1, PD_MODE_SAME, activation);
	//subst->srcOps.push_back(split);
	subst->srcOps.push_back(conv1);
	subst->srcOps.push_back(conv2);
	//OpX* enlarge = subst->create_enlarge(w1, w2, false/*isSrc*/);
	OpX* concat = subst->create_concat(0/*axis*/, 4/*dim*/, w1,
		w2, false/*isSrc*/);
	OpX* conv3 = subst->create_conv2d(input, concat->outputs[0], 1, 1,
		PD_MODE_SAME, activation, false/*isSrc*/);
	OpX* split2 = subst->create_split(conv3->outputs[0], 1/*axis*/, 2, false/*isSrc*/);
	//subst->dstOps.push_back(enlarge);
	subst->dstOps.push_back(concat);
	subst->dstOps.push_back(conv3);
	subst->dstOps.push_back(split2);
	subst->map_output(conv1->outputs[0], split2->outputs[0]);
	subst->map_output(conv2->outputs[0], split2->outputs[1]);
	return subst;
}

GraphXfer* GraphXfer::create_merge_group_convs(Model* model,
                                               int strideH,
                                               int strideW,
                                               ActiMode activation)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX w = subst->new_tensor();
  OpX* conv1 = subst->create_conv2d(input, w, strideH, strideW, PD_MODE_SAME, activation);
  subst->srcOps.push_back(conv1);
  OpX* merge = subst->create_merge_gconv(w, 2/*count*/, false/*isSrc*/);
  OpX* conv2 = subst->create_conv2d(input, merge->outputs[0], strideH, strideW, PD_MODE_SAME, activation, false/*isSrc*/);
  subst->dstOps.push_back(merge);
  subst->dstOps.push_back(conv2);
  subst->map_output(conv1->outputs[0], conv2->outputs[0]);
  return subst;
}

GraphXfer* create_merge_seperable_convs(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input1 = subst->new_tensor();
  TensorX input2 = subst->new_tensor();
  TensorX w1 = subst->new_tensor();
  TensorX w2 = subst->new_tensor();
  TensorX w3 = subst->new_tensor();
  TensorX w4 = subst->new_tensor();
  OpX* conv1 = subst->create_conv2d(input1, w1, 1, 1, PD_MODE_SAME,
                                    AC_MODE_NONE);
  OpX* conv2 = subst->create_conv2d(input2, w2, 1, 1, PD_MODE_SAME,
                                    AC_MODE_NONE);
  OpX* conv3 = subst->create_conv2d(conv1->outputs[0], w3, 1, 1,
                                    PD_MODE_SAME, AC_MODE_NONE);
  OpX* conv4 = subst->create_conv2d(conv2->outputs[0], w4, 1, 1,
                                    PD_MODE_SAME, AC_MODE_NONE);
  OpX* add = subst->create_element(conv3->outputs[0], conv4->outputs[0],
                                   OP_EW_ADD);
  OpX* concatIn = subst->create_concat(1/*axis*/, 4/*dim*/, input1, input2, false/*isSrc*/);
  OpX* concat1 = subst->create_concat(0/*axis*/, 4/*dim*/, w1, w2, false/*isSrc*/);
  OpX* concat2 = subst->create_concat(1/*axis*/, 4/*dim*/, w3, w4, false/*isSrc*/);
  OpX* conv5 = subst->create_conv2d(concatIn->outputs[0], concat1->outputs[0], 1, 1,
                                    PD_MODE_SAME, AC_MODE_NONE, false/*isSrc*/);
  OpX* conv6 = subst->create_conv2d(conv5->outputs[0], concat2->outputs[0], 1, 1,
                                    PD_MODE_SAME,AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(add->outputs[0], conv6->outputs[0]);
  subst->srcOps.push_back(conv1);
  subst->srcOps.push_back(conv2);
  subst->srcOps.push_back(conv3);
  subst->srcOps.push_back(conv4);
  subst->srcOps.push_back(add);
  subst->dstOps.push_back(concatIn);
  subst->dstOps.push_back(concat1);
  subst->dstOps.push_back(concat2);
  subst->dstOps.push_back(conv5);
  subst->dstOps.push_back(conv6);
  return subst;
}

bool get_parameter_from_pb(const GraphSubst::Operator& pbOp,
                           PMParameter pm,
                           int &value)
{
  for (int i = 0; i < pbOp.para_size(); i++)
    if (pbOp.para(i).key() == pm) {
      value = pbOp.para(i).value();
      return true;
    }
  return false;  
}

void GraphXfer::create_operator_from_pb(const GraphSubst::Operator& pbOp,
                                        std::map<int, TensorX>& mappedInputs,
                                        bool isSrcOp)
{
  // Step 1: create inputs
  TensorX inputs[MAX_NUM_INPUTS];
  assert(pbOp.input_size() <= MAX_NUM_INPUTS);
  for (int i = 0; i < pbOp.input_size(); i++) {
    const GraphSubst::Tensor& tensor = pbOp.input(i);
    if (tensor.opid() < 0) {
      int opId = tensor.opid();
      if (mappedInputs.find(opId) == mappedInputs.end()) {
        mappedInputs[opId] = new_tensor();
        assert(isSrcOp); // assert we are still in the src graph
      }
      inputs[i] = mappedInputs[opId];
    } else {
      int opId = tensor.opid();
      int tsId = tensor.tsid();
      if (isSrcOp)
        inputs[i] = srcOps[opId]->outputs[tsId];
      else
        inputs[i] = dstOps[opId]->outputs[tsId];
    }
  }
  // Step 2: create op
  OpType type = (OpType) pbOp.type();
  OpX* opx = NULL;
  switch (type) {
    case OP_CONV2D:
    {
      assert(pbOp.input_size() == 2);
      int strideH, strideW, padding, activation;
      //get_parameter_from_pb(pbOp, PM_KERNEL_H, kernelH);
      //get_parameter_from_pb(pbOp, PM_KERNEL_W, kernelW);
      assert(get_parameter_from_pb(pbOp, PM_STRIDE_H, strideH));
      assert(get_parameter_from_pb(pbOp, PM_STRIDE_W, strideW));
      assert(get_parameter_from_pb(pbOp, PM_PAD, padding));
      assert(get_parameter_from_pb(pbOp, PM_ACTI, activation));
      opx = create_conv2d(inputs[0], inputs[1], strideH, strideW,
          (PaddingMode) padding, (ActiMode) activation, isSrcOp);
      break;
    }
    case OP_CONCAT:
    {
      int numDim, axis;
      assert(get_parameter_from_pb(pbOp, PM_AXIS, axis));
      assert(get_parameter_from_pb(pbOp, PM_NUMDIM, numDim));
      opx = create_concat(axis, numDim, pbOp.input_size(), inputs, isSrcOp);
      break;
    }
    case OP_EW_ADD:
    case OP_EW_MUL:
    {
      assert(pbOp.input_size() == 2);
      opx = create_element(inputs[0], inputs[1], type, isSrcOp);
      break;
    }
    case OP_SPLIT:
    {
      assert(pbOp.input_size() == 1);
      int numOutputs, axis;
      assert(get_parameter_from_pb(pbOp, PM_AXIS, axis));
      assert(get_parameter_from_pb(pbOp, PM_NUM_OUTPUTS, numOutputs));
      opx = create_split(inputs[0], axis, numOutputs, isSrcOp);
      break;
    }
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
    {
      assert(pbOp.input_size() == 1);
      opx = create_activation(inputs[0], type);
      break;
    }
    case OP_MUL:
    {
      assert(pbOp.input_size() == 2);
      opx = create_mul(inputs[0], inputs[1]);
      break;
    }
    case OP_ENLARGE:
    {
      assert(pbOp.input_size() == 2);
      //int kernelH, kernelW;
      //assert(get_parameter_from_pb(pbOp, PM_KERNEL_H, kernelH));
      //assert(get_parameter_from_pb(pbOp, PM_KERNEL_W, kernelW));
      opx = create_enlarge(inputs[0], inputs[1], isSrcOp);
      break;
    }
    case OP_MATMUL:
    {
      assert(pbOp.input_size() == 2);
      int activation;
      assert(get_parameter_from_pb(pbOp, PM_ACTI, activation));
      opx = create_matmul(inputs[0], inputs[1], (ActiMode) activation);
      break;
    }
    case OP_TRANSPOSE:
    {
      assert(pbOp.input_size() == 1);
      int numDim, permIdx, perm[MAX_DIM], shuffle;
      assert(get_parameter_from_pb(pbOp, PM_NUMDIM, numDim));
      assert(get_parameter_from_pb(pbOp, PM_PERM, permIdx));
      assert(get_parameter_from_pb(pbOp, PM_OUTSHUFFLE, shuffle));
      for (int i = numDim-1; i >=0; i--) {
        perm[i] = permIdx % numDim;
        permIdx = permIdx / numDim;
      }
      assert(permIdx == 0);
      for (int i = 0; i < numDim; i++)
        for (int j = i + 1; j < numDim; j++)
          assert(perm[i] != perm[j]);
      opx = create_transpose(inputs[0], numDim, perm, shuffle);
      break;
    }
    case OP_POOL2D_MAX:
    case OP_POOL2D_AVG:
    case OP_BATCHNORM:
    default:
    {
      assert(false);
    }
  }
  assert(opx != NULL);
  if (isSrcOp)
    srcOps.push_back(opx);
  else
    dstOps.push_back(opx);
}

void GraphXfer::load_graph_xfer_from_pb_file(Model* model,
                                             std::vector<GraphXfer*>& xfers,
                                             std::string filename)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  GraphSubst::RuleCollection collection;
  std::fstream input(filename, ios::in);
  assert(collection.ParseFromIstream(&input));
  //printf("Number of generated substitutions = %d\n", collection.rule_size());
  for (int i = 0; i < collection.rule_size(); i++) {
    const GraphSubst::Rule& rule = collection.rule(i);
    std::map<int, TensorX> mappedInputs;
    GraphXfer* subst = new GraphXfer(model);
    for (int j = 0; j < rule.srcop_size(); j++)
      subst->create_operator_from_pb(rule.srcop(j), mappedInputs, true);
    for (int j = 0; j < rule.dstop_size(); j++)
      subst->create_operator_from_pb(rule.dstop(j), mappedInputs, false);
    for (int j = 0; j < rule.mappedoutput_size(); j++) {
      const GraphSubst::MapOutput& mapOutput = rule.mappedoutput(j);
      int srcOpId = mapOutput.srcopid();
      int dstOpId = mapOutput.dstopid();
      int srcTsId = mapOutput.srctsid();
      int dstTsId = mapOutput.dsttsid();
      assert(srcOpId < (int)subst->srcOps.size());
      assert(dstOpId < (int)subst->dstOps.size());
      assert(srcTsId < (int)subst->srcOps[srcOpId]->outputs.size());
      assert(dstTsId < (int)subst->dstOps[dstOpId]->outputs.size());
      subst->map_output(subst->srcOps[srcOpId]->outputs[srcTsId],
                        subst->dstOps[dstOpId]->outputs[dstTsId]);
    }
    xfers.push_back(subst);
  }
}

// Helper functions
TNParameter to_tn_parameter(bool isInput, int n)
{
  switch (n) {
    case 0: return isInput ? IN_0 : OU_0;
    case 1: return isInput ? IN_1 : OU_1;
    case 2: return isInput ? IN_2 : OU_2;
    case 3: return isInput ? IN_3 : OU_3;
    case 4: return isInput ? IN_4 : OU_4;
    case 5: return isInput ? IN_5 : OU_5;
    default:
      assert(false);
  }
  assert(false);
}

DIMParameter to_dim_parameter(int n)
{
  switch (n) {
    case 0: return DIM_0;
    case 1: return DIM_1;
    case 2: return DIM_2;
    case 3: return DIM_3;
    default:
      assert(false);
  }
  assert(false);
}

PMConstraint::PMConstraint(Compare c, PMParameter p, int v)
: comp(c), para(p), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p, DIMParameter d, int v)
: singlePara(true), comp(c), para1(p), dim1(d), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p1, DIMParameter d1,
                           TNParameter p2, DIMParameter d2)
: singlePara(false), comp(c), para1(p1), para2(p2), dim1(d1), dim2(d2) {}

Tensor TensorX::to_tensor(const GraphXfer* xfer) const
{
  if (op != NULL) {
    assert(op->mapOp.ptr != NULL);
    return op->mapOp.ptr->outputs[idx];
  } else {
    std::multimap<int, std::pair<Op, int> >::const_iterator it;
    it = xfer->mappedInputs.find(idx);
    assert(it != xfer->mappedInputs.end());
    Op op = it->second.first;
    int outIdx = it->second.second;
    return op.ptr->outputs[outIdx];
  }
}

//void add_out_edges(TensorX e)
//{
//  if (e.op != NULL) e.op->numOutEdges ++;
//}

OpX::OpX(const OpX& _op)
: type(_op.type), mapOp(_op.mapOp), inputs(_op.inputs), outputs(_op.outputs),
  pmConstraints(_op.pmConstraints), tnConstraints(_op.tnConstraints)
{}

OpX::OpX(OpType _type, TensorX in1, int numOutputs)
: type(_type)
{
  inputs.push_back(in1);
  switch (type) {
    case OP_RESHAPE:
    case OP_TRANSPOSE:
    case OP_RELU:
    case OP_TANH:
    case OP_SIGMOID:
    case OP_MERGE_GCONV:
    {
      TensorX out(this, 0);
      outputs.push_back(out);
      break;
    }
    case OP_SPLIT:
      for (int i = 0; i < numOutputs; i++) {
        TensorX out(this, i);
        outputs.push_back(out);
      }
      break;
    default:
      assert(false);
  }
}

OpX::OpX(OpType _type, TensorX in1, TensorX in2)
: type(_type)
{
  inputs.push_back(in1);
  inputs.push_back(in2);
  TensorX out(this, 0);
  switch (type) {
    case OP_CONV2D:
    case OP_EW_ADD:
    case OP_EW_MUL:
    case OP_POOL2D_AVG:
    case OP_CONCAT:
    case OP_MATMUL:
    case OP_MUL:
    case OP_ENLARGE:
      outputs.push_back(out);
      break;
    default:
      assert(false);
  }
}

OpX::OpX(OpType _type, TensorX in1, TensorX in2, TensorX in3, TensorX in4, TensorX in5)
: type(_type)
{
  inputs.push_back(in1);
  inputs.push_back(in2);
  inputs.push_back(in3);
  inputs.push_back(in4);
  inputs.push_back(in5);
  TensorX out(this, 0);
  switch (type) {
    case OP_BATCHNORM:
      outputs.push_back(out);
      break;
    default:
      assert(false);
  }
}


OpX::OpX(OpType _type, int n, TensorX* ins)
: type(_type)
{
  for (int i = 0; i < n; i++) {
    inputs.push_back(ins[i]);
  }
  TensorX out(this, 0);
  outputs.push_back(out);
}

bool OpX::add_pm_constraint(Compare comp, PMParameter para, int value)
{
  PMConstraint pmc(comp, para, value);
  pmConstraints.push_back(pmc);
  return true;
}

bool OpX::add_input_constraint(Compare comp, TNParameter para,
                               DIMParameter dim, int value)
{
  TNConstraint tnc(comp, para, dim, value);
  tnConstraints.push_back(tnc);
  return true;
}

bool OpX::add_input_constraint(Compare comp,
                               TNParameter para1, DIMParameter dim1,
                               TNParameter para2, DIMParameter dim2)
{
  TNConstraint tnc(comp, para1, dim1, para2, dim2);
  tnConstraints.push_back(tnc);
  return true;
}

bool OpX::get_pm_constraint(PMParameter para, int& value) const
{
  for (size_t i = 0; i < pmConstraints.size(); i++)
    if ((pmConstraints[i].comp == COMPARE_EQ)
    && (pmConstraints[i].para == para)) {
      value = pmConstraints[i].value;
      return true;
    }
  return false;
}

bool SrcOp::add_constraint(Compare comp, PMParameter para, int value)
{
  PMConstraint ooc(comp, para, value);
  constraints.push_back(ooc);
  return true;
}

bool SrcOp::match(Op op)
{
  if (op.guid == 0) return false;
  if (type != OP_ANY && type != op.ptr->type)
    return false;
  bool pass = true;
  for (size_t i = 0; i < constraints.size(); i++) {
    PMConstraint ooc = constraints[i];
    int actValue = 0;
    assert(op.ptr->get_int_parameter(ooc.para, &actValue));
    switch (ooc.comp) {
      case COMPARE_EQ:
        if (actValue != ooc.value) pass = false;
        break;
      case COMPARE_NE:
        if (actValue == ooc.value) pass = false;
        break;
      case COMPARE_LT:
        if (actValue >= ooc.value) pass = false;
        break;
      case COMPARE_LE:
        if (actValue > ooc.value) pass = false;
        break;
      case COMPARE_GT:
        if (actValue <= ooc.value) pass = false;
        break;
      case COMPARE_GE:
        if (actValue < ooc.value) pass = false;
        break;
      default:
        assert(false);
    }
  }
  return pass;
}

/*
SrcEdge::SrcEdge(int _idx, SrcOp* _op)
: idx(_idx), op(_op)
{}

DstEdge::DstEdge(int _idx, DstOp* _op)
: idx(_idx), op(_op)
{}
*/

GraphXfer::GraphXfer(Model* _model)
: model(_model), tensorId(10)
{}

OpX* GraphXfer::create_activation(TensorX input, OpType type, bool isSrcOp)
{
  OpX* activation = new OpX(type, input);
  return activation;
}

OpX* GraphXfer::create_conv2d(TensorX input, TensorX weight,
                              //int kernelH, int kernelW,
                              int strideH, int strideW,
                              PaddingMode padding,
                              ActiMode activation,
                              bool isSrcOp)
{
  OpX* conv = new OpX(OP_CONV2D, input, weight);
  //conv->add_pm_constraint(COMPARE_EQ, PM_KERNEL_H, kernelH);
  //conv->add_pm_constraint(COMPARE_EQ, PM_KERNEL_W, kernelW);
  conv->add_pm_constraint(COMPARE_EQ, PM_STRIDE_H, strideH);
  conv->add_pm_constraint(COMPARE_EQ, PM_STRIDE_W, strideW);
  conv->add_pm_constraint(COMPARE_EQ, PM_PAD, padding);
  conv->add_pm_constraint(COMPARE_EQ, PM_ACTI, activation);
  //conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_2, kernelH);
  //conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_3, kernelW);
  // The following is no longer true because of group conv
  //conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_1, IN_0, DIM_1);
  return conv;
}

OpX* GraphXfer::create_batchnorm(TensorX input, TensorX scale,
                                 TensorX bias, TensorX mean,
                                 TensorX var)
{
  OpX* batch = new OpX(OP_BATCHNORM, input, scale, bias, mean, var);
  return batch;
}

OpX* GraphXfer::create_element(TensorX input0, TensorX input1,
                               OpType type, bool isSrcOp)
{
  OpX* element = new OpX(type, input0, input1);
  return element;
}

OpX* GraphXfer::create_pool2d_avg(TensorX input, TensorX weight,
                                  int strideH, int strideW,
                                  PaddingMode padding,
                                  ActiMode activation,
                                  bool isSrcOp)
{
  OpX* pool = new OpX(OP_POOL2D_AVG, input, weight);
  pool->add_pm_constraint(COMPARE_EQ, PM_STRIDE_H, strideH);
  pool->add_pm_constraint(COMPARE_EQ, PM_STRIDE_W, strideW);
  pool->add_pm_constraint(COMPARE_EQ, PM_PAD, padding);
  pool->add_pm_constraint(COMPARE_EQ, PM_ACTI, activation);
  pool->add_input_constraint(COMPARE_EQ, IN_1, DIM_0, IN_0, DIM_1);
  return pool;
}

OpX* GraphXfer::create_matmul(TensorX input, TensorX weight,
                              ActiMode activation,
                              bool isSrcOp)
{
  OpX* matmul = new OpX(OP_MATMUL, input, weight);
  matmul->add_pm_constraint(COMPARE_EQ, PM_ACTI, activation);
  matmul->add_input_constraint(COMPARE_EQ, IN_1, DIM_0, IN_0, DIM_1);
  return matmul;
}

OpX* GraphXfer::create_mul(TensorX x, TensorX y, bool isSrcOp)
{
  OpX* mul = new OpX(OP_MUL, x, y);
  mul->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND, 0);
  return mul;
}

OpX* GraphXfer::create_transpose(TensorX input, int numDim, int* perm,
                                 int shuffle)
{
  OpX* transpose = new OpX(OP_TRANSPOSE, input);
  int permIdx = 0;
  for (int i = 0; i < numDim; i++)
    permIdx = permIdx * numDim + perm[i];
  transpose->add_pm_constraint(COMPARE_EQ, PM_PERM, permIdx);
  transpose->add_pm_constraint(COMPARE_EQ, PM_OUTSHUFFLE, shuffle);
  transpose->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND, numDim);
  return transpose;
}

OpX* GraphXfer::create_enlarge(TensorX w1, TensorX w2, bool isSrcOp)
{
  OpX* enlarge = new OpX(OP_ENLARGE, w1, w2);
  //enlarge->add_pm_constraint(COMPARE_EQ, PM_KERNEL_H, kernelH);
  //enlarge->add_pm_constraint(COMPARE_EQ, PM_KERNEL_W, kernelW);
  enlarge->add_input_constraint(COMPARE_LE, IN_0, DIM_2, IN_1, DIM_2);
  enlarge->add_input_constraint(COMPARE_LE, IN_0, DIM_3, IN_1, DIM_3);
  return enlarge;
}

//no input constraint, which is different from create_enlarge
OpX* GraphXfer::create_enlarge_a(TensorX w1, TensorX w2, bool isSrcOp)
{
	OpX* enlarge = new OpX(OP_ENLARGE, w1, w2);
	//enlarge->add_pm_constraint(COMPARE_EQ, PM_KERNEL_H, kernelH);
	//enlarge->add_pm_constraint(COMPARE_EQ, PM_KERNEL_W, kernelW);
	//enlarge->add_input_constraint(COMPARE_LT, IN_0, DIM_2, IN_1, DIM_2);
	//enlarge->add_input_constraint(COMPARE_LE, IN_0, DIM_3, IN_1, DIM_3);
	return enlarge;
}

OpX* GraphXfer::create_merge_gconv(TensorX w, int count, bool isSrcOp)
{
  OpX* merge = new OpX(OP_MERGE_GCONV, w);
  merge->add_pm_constraint(COMPARE_EQ, PM_MERGE_GCONV_COUNT, count);
  return merge;
}

OpX* GraphXfer::create_concat(int axis, int numDim, TensorX in1, TensorX in2, bool isSrcOp)
{
  TensorX ins[2];
  ins[0] = in1; ins[1] = in2;
  return create_concat(axis, numDim, 2, ins, isSrcOp);
}

OpX* GraphXfer::create_concat(int axis, int numDim, int n, TensorX* ins, bool isSrcOp)
{
  OpX* concat = new OpX(OP_CONCAT, n, ins);
  concat->add_pm_constraint(COMPARE_EQ, PM_AXIS, axis);
  concat->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND, numDim);
  for (int i = 1; i < n; i++) {
    TNParameter in_i = to_tn_parameter(true/*is_input*/, i);
    concat->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND,
                                 in_i, DIM_ND);
    for (int j = 0; j < numDim; j++) {
      DIMParameter dim_j = to_dim_parameter(j);
      if (j != axis)
        concat->add_input_constraint(COMPARE_EQ, IN_0, dim_j,
                                     in_i, dim_j);
    }
  }
  return concat;
}

OpX* GraphXfer::create_split(TensorX input, int axis, int n, bool isSrcOp)
{
  OpX* split = new OpX(OP_SPLIT, input, n);
  split->add_pm_constraint(COMPARE_EQ, PM_AXIS, axis);
  return split;
}

TensorX GraphXfer::new_tensor(void)
{
  TensorX t;
  t.op = NULL;
  t.idx = tensorId++;
  return t;
}

bool GraphXfer::map_output(TensorX src, TensorX dst)
{
  mappedOutputs[src] = dst;
  return true;
}

//void GraphXfer::add_src_op(SrcOp* src)
//{
//  srcInEdges[src];
//  srcOutEdges[src];
//  srcOps.push_back(src);
//}
//
//void GraphXfer::add_dst_op(DstOp* dst)
//{
//  dstInEdges[dst];
//  dstOutEdges[dst];
//  dstOps.push_back(dst);
//}

//void GraphXfer::add_src_edge(SrcOp* srcOp, SrcOp* dstOp, int srcIdx, int dstIdx)
//{
//  SubEdge<SrcOp> e(srcOp, dstOp, srcIdx, dstIdx);
//  srcInEdges[dstOp].insert(e);
//  srcOutEdges[srcOp].insert(e);
//}

//void GraphXfer::add_dst_edge(DstOp* srcOp, DstOp* dstOp, int srcIdx, int dstIdx)
//{
//  SubEdge<DstOp> e(srcOp, dstOp, srcIdx, dstIdx);
//  dstInEdges[dstOp].insert(e);
//  dstOutEdges[srcOp].insert(e);
//}

//bool GraphXfer::add_constraint(Compare comp,
//                               SrcOp* src, PMParameter srcPara,
//                               SrcOp* dst, PMParameter dstPara)
//{
//  TwoOpConstraint gc(comp, src, srcPara, dst, dstPara);
//  constraints.push_back(gc);
//  return true;
//}

//bool GraphXfer::map_input(SrcOp* src, DstOp* dst)
//{
//  assert(src->mapInput == NULL);
//  assert(dst->mapInput == NULL);
//  src->mapInput = dst;
//  dst->mapInput = src;
//  return true;
//}

//bool GraphXfer::map_output(SrcOp* src, DstOp* dst)
//{
//  assert(src->mapOutput == NULL);
//  assert(dst->mapOutput == NULL);
//  src->mapOutput = dst;
//  dst->mapOutput = src;
//  return true;
//}

bool GraphXfer::can_match(OpX* srcOp, Op op, Graph* graph)
{
  if (srcOp->type != op.ptr->type) return false;
  // check num input tensors
  if ((int)srcOp->inputs.size() != op.ptr->numInputs) return false;
  // check pmConstraints
  for (size_t i = 0; i < srcOp->pmConstraints.size(); i++) {
    PMConstraint pmc = srcOp->pmConstraints[i];
    int actValue = 0;
    assert(op.ptr->get_int_parameter(pmc.para, &actValue));
    //printf("pmc[%d] para(%d) comp(%d) value(%d) actValue(%d)\n",
    //       i, pmc.para, pmc.comp, pmc.value, actValue);
    switch (pmc.comp) {
      case COMPARE_EQ:
      {
        if (actValue != pmc.value) return false;
        break;
      }
      case COMPARE_NE:
      {
        if (actValue == pmc.value) return false;
        break;
      }
      case COMPARE_LT:
      {
        if (actValue >= pmc.value) return false;
        break;
      }
      case COMPARE_LE:
      {
        if (actValue > pmc.value) return false;
        break;
      }
      case COMPARE_GT:
      {
        if (actValue <= pmc.value) return false;
        break;
      }
      case COMPARE_GE:
      {
        if (actValue < pmc.value) return false;
        break;
      }
      default:
        assert(false);
    }
  }
  
  /*if (op.guid == 111)
	  printf("checking input\n");*/
  
  // check inputs
  std::map<int, std::pair<Op, int> > newMapInputs;
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // input tensor
      std::multimap<int, std::pair<Op, int> >::const_iterator it;
      it = mappedInputs.find(in.idx);
      if (it != mappedInputs.end()) {
        Op mappedOp = it->second.first;
        int mappedIdx = it->second.second;
        if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
          return false;
      } else {
        std::map<int, std::pair<Op, int> >::const_iterator newit;
        newit = newMapInputs.find(in.idx);
        if (newit != newMapInputs.end()) {
          Op mappedOp = newit->second.first;
          int mappedIdx = newit->second.second;
          if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
            return false;
        } else {
          std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
          std::set<Edge, EdgeCompare>::const_iterator it2;
          for (it2 = list.begin(); it2 != list.end(); it2++) {
            Edge e = *it2;
            if (e.dstIdx == (int)i) {
              newMapInputs.insert(std::make_pair(in.idx,
                                      std::make_pair(e.srcOp, e.srcIdx)));
            }
          }
        }
        // Do nothing when we check the match
        /* mapped in.idx to an op
        std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
        std::set<Edge, EdgeCompare>::const_iterator it2;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          Edge e = *it2;
          if (e.dstIdx == i)
            mappedInputs[in.idx] = std::make_pair(e.srcOp, e.srcIdx);
        }*/
      }
    } else {
      // intermediate tensor

		/*if (op.guid == 111)
			printf("intermediate input\n");*/

      assert(in.op->mapOp.ptr != NULL);
      if (!(graph->has_edge(in.op->mapOp, op, in.idx, i)))
        return false;
    }
  }
  
  /*if (op.guid == 111)
	  printf("success input\n");*/
  
  // check tnConstraints
  for (size_t i = 0; i < srcOp->tnConstraints.size(); i++) {
    TNConstraint tnc = srcOp->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op.ptr->get_input_parameter(tnc.para2, tnc.dim2, &expValue));
    }
    switch (tnc.comp) {
      case COMPARE_EQ:
      {
        if (actValue != expValue) return false;
        break;
      }
      case COMPARE_NE:
      {
        if (actValue == expValue) return false;
        break;
      }
      case COMPARE_LT:
      {
        if (actValue >= expValue) return false;
        break;
      }
      case COMPARE_LE:
      {
        if (actValue > expValue) return false;
        break;
      }
      case COMPARE_GT:
      {
        if (actValue <= expValue) return false;
        break;
      }
      case COMPARE_GE:
      {
        if (actValue < expValue) return false;
        break;
      }
      default:
        assert(false);
    }
  }
  return true;
}

void GraphXfer::match(OpX* srcOp, Op op, Graph* graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputs
      std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
      std::set<Edge, EdgeCompare>::const_iterator it2;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        Edge e = *it2;
        if (e.dstIdx == (int)i) {
          mappedInputs.insert(std::make_pair(in.idx,
                                  std::make_pair(e.srcOp, e.srcIdx)));
        }
      }
    }
  }
  // Map srcOp to Op
  srcOp->mapOp = op;
  mappedOps[op] = srcOp;
}

void GraphXfer::unmatch(OpX* srcOp, Op op, Graph* graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputsa
      std::multimap<int, std::pair<Op, int> >::iterator it;
      it = mappedInputs.find(in.idx);
      mappedInputs.erase(it);
    }
  }
  // Unmap op
  mappedOps.erase(op);
  srcOp->mapOp.guid = 0;
  srcOp->mapOp.ptr = NULL;
}


//---------------------THIS FUNCTION FOR ALL SEARCH ALGORITHMS TO DEAL WITH symmetric substitution rules----------------------//
//guid1 is the guid of the first mapped op, guid2 is the guid of the second mapped op
//substtype is the type of the substitution rule used in this step
//if it returns true, then this substitution is ok, not redundant; otherwise, this substitution would be redundant
bool is_symmetric_rule(int substtype) {
	//int to_add[] = { 18, 19, 94, 97, 101, 105, 115, 119, 122, 123, 124, 129, 150, 151, 152, 154, 155 };
	int to_add[] = { 18, 19, 94, 101, 105, 115, 119, 123, 129, 151, 154 };

	int to_add_num = 11;
	std::set<int> symmetric_rules(to_add, to_add + to_add_num);
	if (symmetric_rules.find(substtype) != symmetric_rules.end()) {
		//if this substtype is symmetric, then it must has at least two src ops
		return true;
	}
	else {
		//the subst rule is not symmetric
		return false;
	}
}


//---------------------THIS FUNCTION FOR ALL SEARCH ALGORITHMS TO DEAL WITH symmetric substitution rules----------------------//
//guid1 is the guid of the first mapped op, guid2 is the guid of the second mapped op
//substtype is the type of the substitution rule used in this step
//if it returns true, then this substitution is ok, not redundant; otherwise, this substitution would be redundant
bool success_for_symmetric_check(const std::vector<OpX*>& srcops, int substtype) {
	//int to_add[] = {18, 19, 94, 97, 101, 105, 115, 119, 122, 123, 124, 129, 150, 151, 152, 154, 155};
	/*int to_add[] = { 18, 19, 94, 101, 105, 115, 119, 123, 129, 151, 154 };

	int to_add_num = 17;
	std::set<int> symmetric_rules (to_add, to_add + to_add_num);*/
	//if (symmetric_rules.find(substtype) != symmetric_rules.end()) {
	if (is_symmetric_rule(substtype)) {
		//if this substtype is symmetric, then it must has at least two src ops
		int guid1 = srcops[0]->mapOp.guid;
		int guid2 = srcops[1]->mapOp.guid;
		if (guid1 < guid2)
			return true;
		else
			return false;
	}
	else {
		//the subst rule is not symmetric
		return true;
	}
}


//---------------------THIS FUNCTION FOR ALL SEARCH ALGORITHMS TO DEAL WITH symmetric substitution rules----------------------//
//true: if already find a symmetric substitution; otherwise false
bool found_symmetric_subst(Graph* graph, const std::vector<OpX*>& srcops, int substtype) {
	//build the subst key
	struct SubstKey my_key;
	my_key.substType = substtype;
	for (size_t i = 0; i < srcops.size(); i++) {
		my_key.srcOp_guids.push_back(srcops[i]->mapOp.guid);
	}

	if (is_symmetric_rule(substtype)) {
		//if the rule is symmetric
		struct SubstKey to_find = my_key;
		size_t tmp_guid = to_find.srcOp_guids[0];
		to_find.srcOp_guids[0] = to_find.srcOp_guids[1];
		to_find.srcOp_guids[1] = tmp_guid;
		if (graph->symmetric_subst_found.find(to_find) != graph->symmetric_subst_found.end()) {
			//already found one
			return true;
		}
		else {
			graph->symmetric_subst_found.insert(my_key);
			return false;
		}
	}
	else
	{
		//if the rule is not symmetric
		return false;
	}
}


//---------------------THIS FUNCTION FOR ALL SEARCH ALGORITHMS TO CHECK OP DICTIONARY----------------------//
//this function create dst_ops and update op_dict as well
//check_symmetric: for sysml, this value is true; for others, this is false
bool GraphXfer::create_dst_ops_update_op_dict(int substtype, Graph* graph) {
	//build the subst key
	struct SubstKey my_key;
	my_key.substType = substtype;
	for (size_t i = 0; i < srcOps.size(); i++) {
		my_key.srcOp_guids.push_back(srcOps[i]->mapOp.guid);
	}


	//struct SubstKey to_find = my_key;
	////bool need_modify = false;
	//if (is_symmetric_rule(substtype)) {
	//	//if this rule is symmetrick
	//	size_t tmp_guid = to_find.srcOp_guids[0];
	//	to_find.srcOp_guids[0] = to_find.srcOp_guids[1];
	//	to_find.srcOp_guids[1] = tmp_guid;
	//	/*printf("guid1: %zu   guid2: %zu\n", to_find.srcOp_guids[0], to_find.srcOp_guids[1]);
	//	assert(model->op_dict.find(to_find) != model->op_dict.end());*/
	//	//need_modify = true;
	//}


	//if (check_symmetric) {
	//	//for sysml functions
	//	if (success_for_symmetric_check(srcOps, substtype) == false) {
	//		size_t tmp_guid = my_key.srcOp_guids[0];
	//		my_key.srcOp_guids[0] = my_key.srcOp_guids[1];
	//		my_key.srcOp_guids[1] = tmp_guid;
	//		assert(model->op_dict.find(my_key) != model->op_dict.end());
	//	}
	//}

	bool pass = true;
	if (model->op_dict.find(my_key) != model->op_dict.end()) {
		//we can use the op in the dict directly
		for (size_t i = 0; i < dstOps.size(); i++) {
			OpX* dstOp = dstOps[i];
			dstOp->mapOp = model->op_dict[my_key][i];
			Op& temOp = dstOp->mapOp;
			temOp.SIStepOrder = graph->subst_history.size();
		}
		return pass;
	}
	else {
		std::vector<OpX*>::const_iterator dstIt;
		//int count = 0;
		//for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
		for (size_t i = 0; i < dstOps.size(); i++) {
			if (pass) {
				OpX* dstOp = dstOps[i];
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
				Op& temOp = dstOp->mapOp;
				temOp.SIStepOrder = graph->subst_history.size();
				temOp.SIidx = i;
			}
		}

		if (pass) {
			//if (model->op_dict.find(to_find) != model->op_dict.end()) {
			//	//if an equivalent subst has been stored
			//	//we need to modify the cost of the dst ops
			//	for (size_t i = 0; i < dstOps.size(); i++) {
			//		OpX* dstOp = dstOps[i];
			//		Op& temOp = dstOp->mapOp;
			//		if ((temOp.ptr != NULL) && (model->op_dict[to_find][i].ptr != NULL)) {
			//			temOp.ptr->runtime = model->op_dict[to_find][i].ptr->runtime;
			//		}
			//	}
			//}

			// we need update op_dict
			for (size_t i = 0; i < dstOps.size(); i++) {
				model->op_dict[my_key].push_back(dstOps[i]->mapOp);
			}
		}
		return pass;
	}

}


//---------------------THIS FUNCTION ONLY FOR SYSML SEARCH ALGORITHMS TO CHECK OP DICTIONARY FOR COST UPDATE----------------------//
//this function create dst_ops and update op_dict as well
//check_symmetric: for sysml, this value is true; for others, this is false
bool GraphXfer::create_dst_ops_update_op_dict_SYS(int substtype, Graph* graph) {
	//build the subst key
	struct SubstKey my_key;
	my_key.substType = substtype;
	for (size_t i = 0; i < srcOps.size(); i++) {
		if (srcOps[i]->mapOp.guid_for_cost != -1)
			my_key.srcOp_guids.push_back((size_t)(srcOps[i]->mapOp.guid_for_cost));
		else
			my_key.srcOp_guids.push_back(srcOps[i]->mapOp.guid);
	}

	struct SubstKey to_find = my_key;
	
	
	//bool need_modify = false;
	//if (is_symmetric_rule(substtype)) {
	//if (success_for_symmetric_check(srcOps, substtype) == false) {
	//	//if this rule is symmetrick and this subst is not in order
	//	size_t tmp_guid = to_find.srcOp_guids[0];
	//	to_find.srcOp_guids[0] = to_find.srcOp_guids[1];
	//	to_find.srcOp_guids[1] = tmp_guid;
	//	/*printf("guid1: %zu   guid2: %zu\n", to_find.srcOp_guids[0], to_find.srcOp_guids[1]);
	//	assert(model->op_dict.find(to_find) != model->op_dict.end());*/
	//	//need_modify = true;
	//}


	//if (check_symmetric) {
	//	//for sysml functions
		//if (success_for_symmetric_check(srcOps, substtype) == false) {
		//	size_t tmp_guid = my_key.srcOp_guids[0];
		//	my_key.srcOp_guids[0] = my_key.srcOp_guids[1];
		//	my_key.srcOp_guids[1] = tmp_guid;
		//	assert(model->op_dict.find(my_key) != model->op_dict.end());
		//}
	//}

	bool pass = true;
	std::vector<OpX*>::const_iterator dstIt;
	//int count = 0;
	//for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
	for (size_t i = 0; i < dstOps.size(); i++) {
		if (pass) {
			OpX* dstOp = dstOps[i];
			pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
			Op& temOp = dstOp->mapOp;
			temOp.SIStepOrder = graph->subst_history.size();
			temOp.SIidx = i;
		}
	}
	if (!pass)
		return pass;


	//now we need modify the runtime of the op and update the op_dict
	//if (model->op_dict.find(my_key) != model->op_dict.end()) {

	//	//we need to modify the cost and the guid_for_cost of the dst ops
	//	for (size_t i = 0; i < dstOps.size(); i++) {
	//		OpX* dstOp = dstOps[i];
	//		Op& temOp = dstOp->mapOp;
	//		temOp.guid_for_cost = model->op_dict[my_key][i].guid;
	//		if ((temOp.ptr != NULL) && (model->op_dict[my_key][i].ptr != NULL))
	//			temOp.ptr->runtime = model->op_dict[my_key][i].ptr->runtime;
	//	}
	//}
	//else 
	if (model->op_dict.find(to_find) != model->op_dict.end()) {
		//we need to modify the cost and the guid_for_cost of the dst ops
		for (size_t i = 0; i < dstOps.size(); i++) {
			OpX* dstOp = dstOps[i];
			Op& temOp = dstOp->mapOp;
			temOp.guid_for_cost = model->op_dict[to_find][i].guid;
			if ((temOp.ptr != NULL) && (model->op_dict[to_find][i].ptr != NULL))
				temOp.ptr->runtime = model->op_dict[to_find][i].ptr->runtime;
		}
	}
	else {
		// we need update op_dict
		for (size_t i = 0; i < dstOps.size(); i++) {
			model->op_dict[to_find].push_back(dstOps[i]->mapOp);
		}
	}

	return pass;

}


//---------------------THIS FUNCTION OF PRINTING GRAPH SUBST HISTORY FOR ALL SEARCH ALGORITHM----------------------//
//this function prints the subst history of a graph
//graph is the Graph whose subst history needs to be print
void print_subst_history_n_cost_subst(Graph* graph) {
	printf("\n        ===== Applied Substitutions    TOTAL COST: %.4lf =====\n", graph->total_cost());
	for (size_t i = 0; i < graph->subst_history.size(); i++) {
		printf("        substitution[%03zu] ||   type: %d ||   cost change: %.4lf\n", i, graph->subst_history[i].substType, graph->subst_history[i].cost_change);
		Graph::GraphSubst subst = graph->subst_history[i];
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
			printf("            srcOp[%zu]: %s    sisteporder: %d    siidx: %d\n  ", j, subst.srcOps[j].to_string().c_str(), subst.srcOps[j].SIStepOrder, subst.srcOps[j].SIidx);
			std::cout << "address: " << subst.srcOps[j].ptr << "runtime: " << subst.srcOps[j].ptr->runtime << '\n';
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
			printf("            dstOp[%zu]: %s    sisteporder: %d    siidx: %d\n", j, subst.dstOps[j].to_string().c_str(), subst.dstOps[j].SIStepOrder, subst.dstOps[j].SIidx);
			std::cout << "address: " << subst.dstOps[j].ptr << "runtime: " << subst.dstOps[j].ptr->runtime << '\n';
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK (WITHOUT PARTIAL ORDER)----------------------//
//substtime: the time used for substitution (generating the new graph)
void GraphXfer::run_sampletrick(int depth, Graph* graph,
	std::deque<SearchTreeNodesample>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType,
	std::set<Op, OpCompare>& affected_range,
	size_t curr_graph_pos, size_t& add_counter, size_t del_counter, size_t& reuse_counter, float& best_step_cost_reduc, bool can_use_reuse, double& substtime)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {
		//we need to check whether this subtitution would be redundant because of the symmetric rules
		/*if (success_for_symmetric_check(srcOps, substType) == false)
			return;*/
		if (found_symmetric_subst(graph, srcOps, substType))
			return;

		// Create dst operators
		bool pass = create_dst_ops_update_op_dict(substType, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		int count = 0;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
				Op& temOp = dstOp->mapOp;
				temOp.SIStepOrder = graph->subst_history.size();
				temOp.SIidx = count++;
			}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}

		//In order to check whether two substs are redundant or not, print the information of substs here

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_order(graph, substType);
		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());


		//store the cost change: newgraph - oldgraph
		newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();

		//print new graph
		//print_subst_history_n_cost_subst(newGraph);


		//candidates.push_back(newGraph);
		SearchTreeNodesample new_searchTreeNode(newGraph);
		candidates.push_back(new_searchTreeNode);
		//store meda infor for later substitution
		store_infor_sampletrick(candidates.back());
		//add edge in the search tree
		std::vector<std::vector<size_t>>& brothers = candidates.at(curr_graph_pos).childrenNode;
		brothers[substType].push_back(add_counter);
		//deal with sampling: open sample result quota; store sample weight
		//candidates.at(curr_graph_pos).child_sample_quota.push_back(0);
		//candidates.at(curr_graph_pos).child_sample_weight.push_back(newGraph->total_cost());
		add_counter = add_counter + 1;
		
		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;
		
	}
	else {
		//if ((depth == 0) && (graph->subst_history.size() != 0)) {
		if ((depth == 0) && (can_use_reuse)) {
			//mapping the first node in the source graph of this substitution
			OpX* srcOp = srcOps[depth];
			std::set<Op, OpCompare>::const_iterator it;
			for (it = affected_range.begin(); it != affected_range.end(); it++) {

				//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
				if (can_match(srcOp, *it, graph)
					&& (mappedOps.find(*it) == mappedOps.end())) {

					/////////////////////////
					//printf("      this affected op can be matched\n");
					/////////////////////////

					Op op = *it;
					// Check mapOutput
					match(srcOp, op, graph);
					run_sampletrick(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, curr_graph_pos, add_counter, del_counter, reuse_counter, best_step_cost_reduc, can_use_reuse, substtime);
					unmatch(srcOp, op, graph);
				}
				
			}

			///////////////////////////////////
			//printf("        ----------------start reuse explored results-------------------\n");
			///////////////////////////////////

			//here we complete the search in the affected range, then start use explored results below
			SearchTreeNodesample& fatherNode = candidates.front();
			std::vector<size_t>& explored_graphs = fatherNode.childrenNode[substType];
			Graph::GraphSubst& last_final_step = graph->subst_history.back();
			SearchTreeNodesample& curr_graph_node = candidates.at(curr_graph_pos);
			for (size_t explored_i = 0; explored_i < explored_graphs.size(); explored_i++)
			{
				size_t tem_pos = explored_graphs[explored_i] - del_counter;
				const SearchTreeNodesample& explored_searchTreeNode = candidates.at(tem_pos);
				//Graph* explored_graph = explored_searchTreeNode.graphPtr;
				//Graph::GraphSubst& explored_final_step = explored_graph->subst_history.back();
				//change this line because the last optimization step is stored in the search tree node
				const Graph::GraphSubst& explored_final_step = explored_searchTreeNode.subst;
				if ((curr_graph_node.meta_mappedOps.find(explored_final_step.srcOps.front()) == curr_graph_node.meta_mappedOps.end()) && (affected_range.find(explored_final_step.srcOps.front()) == affected_range.end())) {
					//if this explored_graph is not in the affected_range, also exits in the current graph, then we can reuse it
					// reuse dst operators
					//explored_searchTreeNode.meta_data.(this);

					//RECORD THE START OF SUBST
					auto start_time = std::chrono::system_clock::now();

					mappedOps = explored_searchTreeNode.meta_mappedOps;
					mappedInputs.clear();
					//update this.srcOps.mapop, this.dstOps.mapop
					for (size_t dst_i = 0; dst_i < dstOps.size(); dst_i++) {
						OpX* tem_dstOp = dstOps[dst_i];
						tem_dstOp->mapOp = explored_final_step.dstOps[dst_i];
						Op& temOp = tem_dstOp->mapOp;
						temOp.SIStepOrder = graph->subst_history.size();
					}
					for (size_t src_i = 0; src_i < srcOps.size(); src_i++) {
						OpX* tem_srcOp = srcOps[src_i];
						tem_srcOp->mapOp = explored_final_step.srcOps[src_i];
					}

					////////////////////////////////print reused step
					//printf("            substType %d\n", substType);
					/*for (size_t j = 0; j < srcOps.size(); j++) {
					printf("            srcOp[%zu]: %s    substOrder:%d  substID:%d \n", j, srcOps[j]->mapOp.to_string().c_str(), srcOps[j]->mapOp.SIStepOrder, srcOps[j]->mapOp.SIidx);
					}
					for (size_t j = 0; j < dstOps.size(); j++) {
					printf("            dstOp[%zu]: %s    substOrder:%d  substID:%d \n", j, dstOps[j]->mapOp.to_string().c_str(), dstOps[j]->mapOp.SIStepOrder, dstOps[j]->mapOp.SIidx);
					}*/
					////////////////////////////////

					//////////////////////////////////////
					//just for debug, check whether the matched subsgraph is valid or not
					/*printf("mappedOps.size() : %zu, srcOps.size() : %zu\n", mappedOps.size(), srcOps.size());
					std::map<Op, OpX*, OpCompare>::const_iterator debug_mop_itt;
					for (debug_mop_itt = mappedOps.begin(); debug_mop_itt != mappedOps.end(); debug_mop_itt++) {
					printf("the op being asserted: guid:%zu  SIStepOrder:%d    SIidx:%d\n", debug_mop_itt->first.guid, debug_mop_itt->first.SIStepOrder, debug_mop_itt->first.SIidx);
					}
					std::map<Op, OpX*, OpCompare>::const_iterator debug_mop_it;
					for (debug_mop_it = mappedOps.begin(); debug_mop_it != mappedOps.end(); debug_mop_it++) {
					printf("the op being asserted: guid:%zu  SIStepOrder:%d    SIidx:%d\n", debug_mop_it->first.guid, debug_mop_it->first.SIStepOrder, debug_mop_it->first.SIidx);
					for (size_t debug_i = 0; debug_i < srcOps.size(); debug_i++) {
					if (srcOps[debug_i] == debug_mop_it->second) {
					printf("the opx in srcops is also in mappedops\n");
					assert(debug_mop_it->second->mapOp.guid == debug_mop_it->first.guid);
					assert(debug_mop_it->second->mapOp.ptr == debug_mop_it->first.ptr);
					assert(debug_mop_it->second->mapOp.SIStepOrder == debug_mop_it->first.SIStepOrder);
					assert(debug_mop_it->second->mapOp.SIidx == debug_mop_it->first.SIidx);
					}
					}
					assert(graph->inEdges.find(debug_mop_it->first) != graph->inEdges.end());
					}
					assert(mappedOps.size() == srcOps.size());
					if (check(graph) == true)
					printf("REDUNDANT!!!!!!!!!!!!!\n");*/
					//////////////////////////////////////

					//check whether this partial sequence is redundant or not
					//if (compNode(last_final_step.srcOps[0]) >= 0) {

					//change the compNode function because of the new partial order
					//if (compNode(last_final_step.biggestNode, explored_final_step.biggestNode) >= 0) {
					//	//need to erase meta infor stored in this graphXfer before leave
					//	mappedInputs.clear();
					//	mappedOps.clear();
					//	for (size_t erase_i = 0; erase_i < srcOps.size(); erase_i++) {
					//		srcOps[erase_i]->mapOp.guid = 0;
					//		srcOps[erase_i]->mapOp.ptr = NULL;
					//	}
					//	continue;
					//}

					//copy the biggest Node of the reused step
					//biggestNode = explored_final_step.biggestNode;

					////////////////////////////////
					//printf("checked successfully!\n");
					////////////////////////////////

					//std::multimap<int, std::pair<Op, int> > mappedInputs need to be updated
					std::multimap<int, std::pair<Op, int> >::const_iterator in_it;
					for (in_it = explored_searchTreeNode.meta_mappedInputs.begin(); in_it != explored_searchTreeNode.meta_mappedInputs.end(); in_it++) {
						if (curr_graph_node.meta_mappedOps.find((*in_it).second.first) != curr_graph_node.meta_mappedOps.end()) {
							//if the input is substituted by the last step of graph
							TensorX srcTen;
							srcTen.op = curr_graph_node.meta_mappedOps[(*in_it).second.first];
							srcTen.idx = (*in_it).second.second;

							////////////////////////////////////////
							//printf("for debug: srcTen.optyp:e %d   idx:%d\n", srcTen.op->type, srcTen.idx); //print infor for debug
							assert(curr_graph_node.meta_mappedOutputs.find(srcTen) != curr_graph_node.meta_mappedOutputs.end());
							////////////////////////////////////////

							TensorX dstTen = curr_graph_node.meta_mappedOutputs[srcTen];
							//for reference:  newGraph->add_edge(dstTen.op->mapOp, it->dstOp, dstTen.idx, it->dstIdx); 
							size_t tem_dst_pos;
							for (tem_dst_pos = 0; tem_dst_pos < curr_graph_node.meta_dstOps.size(); tem_dst_pos++) {
								if (curr_graph_node.meta_dstOps[tem_dst_pos] == dstTen.op)
									break;
							}
							mappedInputs.insert(std::make_pair(in_it->first, std::make_pair(last_final_step.dstOps[tem_dst_pos], dstTen.idx)));
						}
						else {
							mappedInputs.insert(*in_it);
						}
					}

					//In order to check whether two substs are redundant or not, print the information of substs here


					//check whether this partial sequence is redundant or not
					//if (check(graph) == true)
					//return; //may need to delete hash_map


					// Generate a new graph by applying xfer rule
					Graph* newGraph = create_new_graph_order(graph, substType);
					// Check that the new graph should not have any loop
					if (newGraph->has_loop()) {
						//printf("Found a new graph with LOOP!!!!\n");
						delete newGraph;

						//need to erase meta infor stored in this graphXfer before leave
						mappedInputs.clear();
						mappedOps.clear();
						for (size_t erase_i = 0; erase_i < srcOps.size(); erase_i++) {
							srcOps[erase_i]->mapOp.guid = 0;
							srcOps[erase_i]->mapOp.ptr = NULL;
						}

						//RECORD THE END OF SUBST
						auto end_time = std::chrono::system_clock::now();
						auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
						double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
						substtime += searchtime;

						continue;
					}
					// TODO: remove me for better performance
					assert(newGraph->check_correctness());

					///////////////////////////
					//newGraph->print_costs();
					///////////////////////////

					//store the cost change: newgraph - oldgraph
					newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();


					//if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
					//if (hashmap.find(newGraph->hash()) == hashmap.end()) { //find a new graph
					//													   /*printf("add candidate!\n");*/
					//	hashmap.insert(newGraph->hash());
					//}
					//else {
					//	//print subst history before delete it
					//	printf("\nTHIS GRAPH IS FILTERED\n");
					//	print_subst_history_n_cost_subst(newGraph);

					//	//the graph has been found
					//	delete newGraph;

					//	//need to erase meta infor stored in this graphXfer before leave
					//	mappedInputs.clear();
					//	mappedOps.clear();
					//	for (size_t erase_i = 0; erase_i < srcOps.size(); erase_i++) {
					//		srcOps[erase_i]->mapOp.guid = 0;
					//		srcOps[erase_i]->mapOp.ptr = NULL;
					//	}

					//	continue;
					//}

					//update best_step_cost_reduc
					/*if ((graph->total_cost() - newGraph->total_cost()) > best_step_cost_reduc) {
					best_step_cost_reduc = graph->total_cost() - newGraph->total_cost();
					printf("current best_step_cost_reduc: %.4lf\n", best_step_cost_reduc);
					}*/


					//candidates.push_back(newGraph);
					SearchTreeNodesample new_searchTreeNode(newGraph);
					candidates.push_back(new_searchTreeNode);
					//store meda infor for later substitution
					store_infor_sampletrick(candidates.back());
					//add edge in the search tree
					std::vector<std::vector<size_t>>& brothers = candidates.at(curr_graph_pos).childrenNode;
					brothers[substType].push_back(add_counter);
					//deal with sampling: open sample result quota; store sample weight
					//candidates.at(curr_graph_pos).child_sample_quota.push_back(0);
					//candidates.at(curr_graph_pos).child_sample_weight.push_back(newGraph->total_cost());

					add_counter = add_counter + 1;

					reuse_counter = reuse_counter + 1;

					//}
					/*} else {
					delete newGraph;
					}*/

					//need to erase infor stored in this graphXfer		
					mappedInputs.clear();
					mappedOps.clear();
					for (size_t erase_i = 0; erase_i < srcOps.size(); erase_i++) {
						srcOps[erase_i]->mapOp.guid = 0;
						srcOps[erase_i]->mapOp.ptr = NULL;
					}

					//RECORD THE END OF SUBST
					auto end_time = std::chrono::system_clock::now();
					auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
					double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
					substtime += searchtime;
				}
			}
		}
		else {
			OpX* srcOp = srcOps[depth];
			std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
			for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
				//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));

				/////////////////////////////
				//printf("        normally checking op:    guid:%zu  SIStepOrder:%d    SIidx:%d\n", it->first.guid, it->first.SIStepOrder, it->first.SIidx);
				/////////////////////////////

				if (can_match(srcOp, it->first, graph)
					&& (mappedOps.find(it->first) == mappedOps.end())) {
					Op op = it->first;
					// Check mapOutput
					match(srcOp, op, graph);
					run_sampletrick(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, curr_graph_pos, add_counter, del_counter, reuse_counter, best_step_cost_reduc, can_use_reuse, substtime);
					unmatch(srcOp, op, graph);
				}
			}
		}
	}
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK----------------------//
void GraphXfer::store_infor_sampletrick(SearchTreeNodesample& search_tree_node) {
	search_tree_node.meta_mappedOps = mappedOps;
	search_tree_node.meta_mappedInputs = mappedInputs;
	search_tree_node.meta_mappedOutputs = mappedOutputs;
	search_tree_node.meta_dstOps = dstOps;

	//to save memory, we need to store the last step in the search tree node
	search_tree_node.subst = search_tree_node.graphPtr->subst_history.back();
}



//---------------------THIS FUNCTION ONLY FOR PRUNE----------------------//
void GraphXfer::run_prune(int depth, Graph* graph,
	//std::deque<Graph*>& candidates,
	std::forward_list<Graph*>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType, double& substtime)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {

		//we need to check whether this subtitution would be redundant because of the symmetric rules
		/*if (success_for_symmetric_check(srcOps, substType) == false)
			return;*/
		if (found_symmetric_subst(graph, srcOps, substType))
			return;


		//check whether this partial sequence is redundant or not
		if (check(graph) == true)
			return; //may need to delete hash_map


					////////////////////////////////
					//printf("checked successfully!\n");
					////////////////////////////////


		// Create dst operators
		bool pass = create_dst_ops_update_op_dict(substType, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		int count = 0;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
				Op& temOp = dstOp->mapOp;
				temOp.SIStepOrder = graph->subst_history.size();
				temOp.SIidx = count++;
			}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();
		
		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_order(graph, substType);
		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		//In order to check whether two substs are redundant or not, print the information of substs here

		//candidates.push_back(newGraph);
		candidates.push_front(newGraph); //because we want to do "depth first now"

		//print the new graph infor
		//print_subst_history_n_cost_subst(newGraph);


		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

	}
	else {
		OpX* srcOp = srcOps[depth];
		std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
		for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
			//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
			if (can_match(srcOp, it->first, graph)
				&& (mappedOps.find(it->first) == mappedOps.end())) {
				Op op = it->first;
				// Check mapOutput
				match(srcOp, op, graph);
				run_prune(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, substtime);
				unmatch(srcOp, op, graph);
			}
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR ENUMERATION----------------------//
void GraphXfer::run_enumeration(int depth, Graph* graph,
	//std::deque<Graph*>& candidates,
	std::forward_list<Graph*>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType, double& substtime)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {

		//we need to check whether this subtitution would be redundant because of the symmetric rules
		/*if (success_for_symmetric_check(srcOps, substType) == false)
		return;*/
		/*if (found_symmetric_subst(graph, srcOps, substType))
			return;*/


		//check whether this partial sequence is redundant or not
		//if (check(graph) == true)
		//	return; //may need to delete hash_map


					////////////////////////////////
					//printf("checked successfully!\n");
					////////////////////////////////


					// Create dst operators
		bool pass = create_dst_ops_update_op_dict_SYS(substType, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		int count = 0;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
		if (pass) {
		OpX* dstOp = *dstIt;
		pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
		Op& temOp = dstOp->mapOp;
		temOp.SIStepOrder = graph->subst_history.size();
		temOp.SIidx = count++;
		}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_sysmlpartition(graph);
		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		//In order to check whether two substs are redundant or not, print the information of substs here

		//candidates.push_back(newGraph);
		candidates.push_front(newGraph); //because we want to do "depth first now"

										 //print the new graph infor
										 //print_subst_history_n_cost_subst(newGraph);


										 //RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

	}
	else {
		OpX* srcOp = srcOps[depth];
		std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
		for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
			//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
			if (can_match(srcOp, it->first, graph)
				&& (mappedOps.find(it->first) == mappedOps.end())) {
				Op op = it->first;
				// Check mapOutput
				match(srcOp, op, graph);
				run_enumeration(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, substtime);
				unmatch(srcOp, op, graph);
			}
		}
	}
}



//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//check whether "to_check" has some ops in common with "last"
//if yes, return true; else return false
bool check_common_ops(const Graph::GraphSubst& last, const Graph::GraphSubst& to_check) {
	//printf("checking common ops!!!!!!!!!!!!!!!!!\n");
	const std::vector<Op>& to_check_srcs = to_check.srcOps;
	const std::vector<Op>& last_srcs = last.srcOps;
	//printf("last src op num: %zu\n", last_srcs.size());
	//printf("reuse src op num: %zu\n", to_check_srcs.size());
	for (size_t i = 0; i < to_check_srcs.size(); i++) {
		for (size_t j = 0; j < last_srcs.size(); j++) {
			//printf("to_check_srcs--i: %zu    last_srcs--j: %zu   \n", i, j);
			if (to_check_srcs[i] == last_srcs[j])
				return true;
		}
	}
	return false;
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//this function deletes all edges related to an op from both inEdges and outEdges
void delete_edges(Graph* graph, const Op& delete_op) {
	const std::set<Edge, EdgeCompare>& list = graph->inEdges.at(delete_op);
	std::set<Edge, EdgeCompare>::const_iterator it;
	for (it = list.begin(); it != list.end(); it++) {
		graph->outEdges[it->srcOp].erase(*it);
	}
	const std::set<Edge, EdgeCompare>& listout = graph->outEdges[delete_op];
	std::set<Edge, EdgeCompare>::const_iterator itout;
	for (itout = listout.begin(); itout != listout.end(); itout++) {
		graph->inEdges[itout->dstOp].erase(*itout);
	}
	//now delete the entry of delete_op itself
	graph->inEdges.erase(delete_op);
	graph->outEdges.erase(delete_op);
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//this function adds all in edges, related to this op
//add_op is the op to add
//curr_node: the node whose child nodes are to be searched now
//reuse_node: the node which is reused now
void add_edges_in(Graph* new_graph, const SearchTreeNode& curr_node, const SearchTreeNode& reuse_node, const Op& add_op) {
	Graph* reuse_graph = reuse_node.graphPtr;
	const std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>& mapout_infor = curr_node.map_output_infor;
	//add new in edges
	const std::set<Edge, EdgeCompare>& list = reuse_graph->inEdges.at(add_op);
	std::set<Edge, EdgeCompare>::const_iterator it;
	//iterate over all new inedges
	for (it = list.begin(); it != list.end(); it++) {
		if (mapout_infor.find(it->srcOp) != mapout_infor.end()) {
			//if the srcop is replaced by last step
			//find the mapped output tensor

			bool find = false;
			const std::set<Edge, EdgeCompare>& tensormap = mapout_infor.at(it->srcOp);
			std::set<Edge, EdgeCompare>::const_iterator tensorit;
			for (tensorit = tensormap.begin(); tensorit != tensormap.end(); tensorit++) {
				if (tensorit->srcIdx == it->srcIdx) {
					//new_graph->add_edge(tensorit->dstOp, it->dstOp, tensorit->dstIdx, it->dstIdx);  //we need to make the op infor in the edge consistent (i.e., the sisteporder)
					new_graph->add_edge(tensorit->dstOp, add_op, tensorit->dstIdx, it->dstIdx);

					////////////////////////////
					//assert(curr_node.graphPtr->inEdges.find(tensorit->dstOp) != curr_node.graphPtr->inEdges.end());
					////////////////////////////

					find = true;
					break;
				}

			}
			assert(find == true);
		}
		else {
			//this srcop is not replaced by last step, i.e., it is still in the newgraph. we need to make the op infor in the edge consistent (i.e., the sisteporder)
			new_graph->add_edge(new_graph->inEdges.find(it->srcOp)->first, add_op, it->srcIdx, it->dstIdx);

			/*new_graph->inEdges[add_op].insert(*it);
			new_graph->outEdges[it->srcOp].insert(*it);*/
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//this function adds all out edges, related to this op
//replace_op is the op to replace
//curr_node: the node whose child nodes are to be searched now
//reuse_node: the node which is reused now
void add_edges_out(Graph* new_graph, const SearchTreeNode& curr_node, const SearchTreeNode& reuse_node, const Op& replace_op, const std::vector<Op>& old_ops) {
	Graph* graph = curr_node.graphPtr;
	const std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>& mapout_infor = reuse_node.map_output_infor;

	if (mapout_infor.find(replace_op) != mapout_infor.end()) {
		//if the output of this old op can be used outside

		//find the outside edge
		const std::set<Edge, EdgeCompare>& list = graph->outEdges[replace_op];
		std::set<Edge, EdgeCompare>::const_iterator it;

		for (it = list.begin(); it != list.end(); it++) {

			bool find = false;
			for (size_t i = 0; i < old_ops.size(); i++) {
				if (old_ops[i] == it->dstOp) {
					find = true;
					break;
				}
			}
			if (!find) {
				//if this edge is an outside edge
				//find the mapped output tensor
				find = false;
				const std::set<Edge, EdgeCompare>& tensormap = mapout_infor.at(it->srcOp);
				std::set<Edge, EdgeCompare>::const_iterator tensorit;
				for (tensorit = tensormap.begin(); tensorit != tensormap.end(); tensorit++) {
					if (tensorit->srcIdx == it->srcIdx) {
						//new_graph->add_edge(tensorit->dstOp, it->dstOp, tensorit->dstIdx, it->dstIdx);
						new_graph->add_edge(new_graph->inEdges.find(tensorit->dstOp)->first, it->dstOp, tensorit->dstIdx, it->dstIdx);
						find = true;
						break;
					}

				}
				assert(find == true);
			}
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//create new graph through reusing directly
//curr_node: the node whose child nodes are to be searched now
//reuse_node: the node which is reused now
Graph* new_graph_reuse(const SearchTreeNode& curr_node, const SearchTreeNode& reuse_node)
{
	Graph* graph = curr_node.graphPtr;
	Graph* reuse_graph = reuse_node.graphPtr;
	Graph* newGraph = new Graph();

	//printf("GENERATE 1\n");

	//update the subst_history
	newGraph->subst_history = graph->subst_history;
	newGraph->subst_history.push_back(reuse_graph->subst_history.back());
	Graph::GraphSubst& reuse_step = newGraph->subst_history.back();
	reuse_step.order = graph->subst_history.size();

	//printf("new last step src op number: %zu\n", reuse_step.srcOps.size());

	//printf("GENERATE 2\n");


	//printf("GENERATE 3\n");

	//update in edges and out edges
	newGraph->inEdges = graph->inEdges;
	newGraph->outEdges = graph->outEdges;

	//update new ops and add corresponding entry in inEdges and outEdges
	for (size_t i = 0; i < reuse_step.dstOps.size(); i++) {
		reuse_step.dstOps[i].SIStepOrder = reuse_step.order;
		newGraph->inEdges[reuse_step.dstOps[i]];
		newGraph->outEdges[reuse_step.dstOps[i]];
	}

	//now we need to add all new in edges
	for (size_t i = 0; i < reuse_step.dstOps.size(); i++) {
		const Op& to_add = reuse_step.dstOps[i];
		add_edges_in(newGraph, curr_node, reuse_node, to_add);
	}

	//printf("GENERATE 4\n");

	//now we add all new out edges
	for (size_t i = 0; i < reuse_step.srcOps.size(); i++) {
		const Op& to_replace = reuse_step.srcOps[i];
		add_edges_out(newGraph, curr_node, reuse_node, to_replace, reuse_step.srcOps);
	}

	//printf("GENERATE 5\n");


	//we need delete all edges, including inedges and outedges, related to this op
	for (size_t i = 0; i < reuse_step.srcOps.size(); i++) {
		const Op& to_replace = reuse_step.srcOps[i];
		delete_edges(newGraph, to_replace);
	}

	//printf("GENERATE 6\n");

	//printf("new last step src op number: %zu   %zu\n", reuse_step.srcOps.size(), newGraph->subst_history.back().srcOps.size());

	//now we complete the graph inedges and outedges update
	return newGraph;
}

//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
void update_mapout(SearchTreeNode& newnode, const SearchTreeNode& reusenode) {
	//we store the output tensor mapping function here	
	//newnode.map_output_infor = reusenode.map_output_infor;
	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opit;
	for (opit = reusenode.map_output_infor.begin(); opit != reusenode.map_output_infor.end(); opit++) {
		const std::set<Edge, EdgeCompare>& list = opit->second;
		std::set<Edge, EdgeCompare>::const_iterator it;
		for (it = list.begin(); it != list.end(); it++) {
			Edge e = *it;
			//e.dstOp.SIStepOrder = e.dstOp.SIStepOrder + 1;
			e.dstOp.SIStepOrder = newnode.graphPtr->subst_history.size() - 1;
			newnode.map_output_infor[opit->first].insert(e);
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
//this function has nothing to do with specific xfer, it directly reuse subst steps of all types
//candidates: stores all search nodes in the search tree
//curr_node_id: the index of current search tree node
void GraphXfer::reuse_all_steps(std::deque<SearchTreeNode>& candidates, int curr_node_id, size_t& add_counter, size_t& reuse_counter, double& substtime) {
	//printf("start reusing................\n");

	SearchTreeNode& curr_node = candidates[curr_node_id];
	/*printf("reuse step 1\n");
	printf("candidate size: %zu\n", candidates.size());
	printf("self id: %d  father id: %d\n", curr_node_id, curr_node.father);*/
	Graph* graph = curr_node.graphPtr;
	const Graph::GraphSubst& last_step = graph->subst_history.back();
	//get the father node of the current search tree node
	const SearchTreeNode& father_node = candidates[curr_node.father];
	//iterate over all steps of the grandpa node of this current node, which are potential subst steps that can be reused
	//father node must has child nodes
	for (size_t i = father_node.first_child; i < father_node.last_child; i++) {
		//printf("try to reuse node %zu\n", i);
		const SearchTreeNode& reuse_node = candidates[i];
		//printf("reuse step 1.5\n");
		Graph* reuse_graph = reuse_node.graphPtr;

		//printf("reuse step 1.6\n");

		//check whether this child node can be reused or not
		if (check_common_ops(last_step, reuse_graph->subst_history.back()))
			continue;

		//printf("reuse step 2\n");

		//here, this child node can be reused
		//we first check the order
		if (compNode(last_step.biggestNode, reuse_graph->subst_history.back().biggestNode) >= 0)
			continue;

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();
		

		//now generate a new graph by reusing
		//printf("the node %zu can be reused\n", i);
		Graph* newGraph = new_graph_reuse(curr_node, reuse_node);

		//printf("GENERATE A NEW GRAPH\n");

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			continue;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());


		//if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
		//if (hashmap.find(newGraph->hash()) == hashmap.end()) {
		/*printf("add candidate!\n");*/
		//printf("reuse one candidate!   %zu\n", candidates.size());
		//hashmap.insert(newGraph->hash());

		//candidates.push_back(newGraph);
		SearchTreeNode new_searchTreeNode(newGraph, curr_node_id);
		update_mapout(new_searchTreeNode, reuse_node);
		candidates.push_back(new_searchTreeNode);
		//store meda infor for later substitution
		//store_infor(candidates.back());

		//add edge in the search tree
		//std::vector<std::vector<size_t>>& brothers = candidates.at(curr_graph_pos).childrenNode;
		//brothers[substType].push_back(add_counter);
		add_counter = add_counter + 1;
		reuse_counter = reuse_counter + 1;

		//print the new graph infor
		//printf("THIS SUBST IS REUSED===================\n");
		//print_subst_history_n_cost_subst(newGraph);

		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

		//printf("the new node %zu   the src op num of it: %zu\n", candidates.size() - 1, candidates[candidates.size() - 1].graphPtr->subst_history.back().srcOps.size());
	}
}


//---------------------THIS FUNCTION ONLY FOR REUSE for the substitution found immediately after a cost-up substitution----------------------//
// this function combine the mapout_infor from the fathernode into the selfnode temporatily
void combine_mapout_infor_from_father(const SearchTreeNode& fathernode, SearchTreeNode& selfnode) {

	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>& self_mapout_infor = selfnode.map_output_infor;
	const std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>& father_mapout_infor = fathernode.map_output_infor;
	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
	for (it = father_mapout_infor.begin(); it != father_mapout_infor.end(); it++) {
		std::set<Edge, EdgeCompare>::const_iterator edge_it;
		for (edge_it = it->second.begin(); edge_it != it->second.end(); edge_it++) {

			const Op& old_op = edge_it->srcOp;
			int old_idx = edge_it->srcIdx;
			const Op& meta_op = edge_it->dstOp;
			int meta_idx = edge_it->dstIdx;
				
			if (self_mapout_infor.find(meta_op) != self_mapout_infor.end()) {
				//if self node replace the output of the meta op
				std::set<Edge, EdgeCompare>::const_iterator meta_edge_it;
				for (meta_edge_it = self_mapout_infor[meta_op].begin(); meta_edge_it != self_mapout_infor[meta_op].end(); meta_edge_it++) {
					if (meta_edge_it->srcIdx == meta_idx) {
						//we find the corresponding tensor mapping
						Edge to_add(old_op, meta_edge_it->dstOp, old_idx, meta_edge_it->dstIdx);
						self_mapout_infor[old_op].insert(to_add);
						break;
					}
				}
				//if we cannot find a mapping tensor in self_mapout_infor, then is tensor will never be used as outer tensor, we do not need to store the mapping
			}
			else {
				// we directly add the mapping to self_mapout_infor, no matter the new tensor exist in selfnode's graph or not
				self_mapout_infor[old_op].insert(*edge_it);
			}
		}
	}
}

//---------------------THIS FUNCTION ONLY FOR REUSE for the substitution found immediately after a cost-up substitution----------------------//
// this function delete the mapout_infor from the fathernode into the selfnode
void delete_mapout_infor_from_father(const SearchTreeNode& fathernode, SearchTreeNode& selfnode) {
	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>& self_mapout_infor = selfnode.map_output_infor;
	const std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>& father_mapout_infor = fathernode.map_output_infor;
	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
	for (it = father_mapout_infor.begin(); it != father_mapout_infor.end(); it++) {
		self_mapout_infor.erase(it->first);
	}
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK WITH TRUE REUSE----------------------//
//this function has nothing to do with specific xfer, it directly reuse subst steps of all types
//candidates: stores all search nodes in the search tree
//curr_node_id: the index of current search tree node, not the raw one
//del_counter: the number of nodes deleted by now
//THIS FUNCTION ALSO DEALS WITH THE CASE WHERE WE FIND SUBST IMMEDIATELY AFTER A COST-UP SUBSTITUTION
void GraphXfer::reuse_all_steps(std::deque<SearchTreeNode>& candidates, int curr_node_id, size_t& add_counter, size_t& reuse_counter, size_t del_counter, double& substtime) {
	//printf("start reusing................\n");

	SearchTreeNode& curr_node = candidates[curr_node_id];
	/*printf("reuse step 1\n");
	printf("candidate size: %zu\n", candidates.size());
	printf("self id: %d  father id: %d\n", curr_node_id, curr_node.father);*/
	Graph* graph = curr_node.graphPtr;
	const Graph::GraphSubst& last_step = graph->subst_history.back();

	size_t length = graph->subst_history.size(); //store the length of subst history if without further search after cost-up substitutions

	//get the father node of the current search tree node
	int father_node_curr_pos = curr_node.father - del_counter;
	size_t range_end = candidates[father_node_curr_pos].last_child - del_counter;

	//if (candidates[father_node_curr_pos].first_child == candidates[father_node_curr_pos].last_child) {
	if (candidates[father_node_curr_pos].further_search) {
		//this curr_node is a substitution after a cost-increasing substitution, then we see its grandfather as	"father"

		//change the value of length
		length = length - 1;

		father_node_curr_pos = candidates[father_node_curr_pos].father - del_counter;
		//we also need to combine the mapout_infor of its real father
		combine_mapout_infor_from_father(candidates[curr_node.father - del_counter], curr_node);

	}
	
	const SearchTreeNode& father_node = candidates[father_node_curr_pos];

	//printf("=================length: %zu   father start: %zu   end: %zu   range_end: %zu    del num: %zu\n", length, candidates[curr_node.father - del_counter].first_child, candidates[curr_node.father - del_counter].last_child, range_end, del_counter);

	//iterate over all steps of the grandpa node of this current node, which are potential subst steps that can be reused
	//father node must has child nodes
	//for (size_t i = father_node.first_child - del_counter; i < father_node.last_child - del_counter; i++) {
	for (size_t i = father_node.first_child - del_counter; i < range_end; i++) {
		//printf("try to reuse node %zu\n", i);
		const SearchTreeNode& reuse_node = candidates[i];
		//printf("reuse step 1.5\n");
		Graph* reuse_graph = reuse_node.graphPtr;

		if (reuse_graph->subst_history.size() > length) {
			//we have iterate over all direct child nodes of father_node
			if (candidates[curr_node.father - del_counter].further_search) {
				i = candidates[curr_node.father - del_counter].first_child - del_counter - 1;
				length++;
				continue;
			}
			else
				break;
		}
			

		//now the reuse node must be the direct child of father_node, therefore can be used by curr_node

		// we can only reuse a node whose history is shorter or whose father is the same
		/*if (((reuse_graph->subst_history.size()) >= (graph->subst_history.size())) && (reuse_node.father != curr_node.father))
			continue;*/

		//printf("reuse step 1.6\n");

		if ((reuse_graph->subst_history.size()) == (graph->subst_history.size())) {
			//in this case, reuse and curr nodes have the same father
			
			//check whether this child node can be reused or not
			if (check_common_ops(last_step, reuse_graph->subst_history.back()))
				continue;
		}
		else {
			//the reuse node is the child of the grandfather of curr node
			if (check_common_ops(candidates[curr_node.father - del_counter].graphPtr->subst_history.back(), reuse_graph->subst_history.back()))
				continue;
			
			//now reuse node can be used by curr node's father

			if (check_common_ops(last_step, reuse_graph->subst_history.back()))
				continue;

		}
		
		//check whether this child node can be reused or not
		/*if (check_common_ops(last_step, reuse_graph->subst_history.back()))
			continue;*/

		//printf("reuse step 2\n");

		//here, this child node can be reused
		//we first check the order
		/*if (compNode(last_step.biggestNode, reuse_graph->subst_history.back().biggestNode) >= 0)
			continue;*/

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		//now generate a new graph by reusing
		//printf("the node %zu can be reused\n", i);
		Graph* newGraph = new_graph_reuse(curr_node, reuse_node);

		//printf("GENERATE A NEW GRAPH\n");

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			continue;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());


		//if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
		//if (hashmap.find(newGraph->hash()) == hashmap.end()) {
		/*printf("add candidate!\n");*/
		//printf("reuse one candidate!   %zu\n", candidates.size());
		//hashmap.insert(newGraph->hash());

		//printf("reuse step 3\n");

		//candidates.push_back(newGraph);
		SearchTreeNode new_searchTreeNode(newGraph, curr_node_id + del_counter);
		new_searchTreeNode.reused = true;
		update_mapout(new_searchTreeNode, reuse_node);
		candidates.push_back(new_searchTreeNode);
		//store meda infor for later substitution
		//store_infor(candidates.back());

		//add edge in the search tree
		//std::vector<std::vector<size_t>>& brothers = candidates.at(curr_graph_pos).childrenNode;
		//brothers[substType].push_back(add_counter);
		add_counter = add_counter + 1;
		reuse_counter = reuse_counter + 1;

		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

		//printf("the new node %zu   the src op num of it: %zu\n", candidates.size() - 1, candidates[candidates.size() - 1].graphPtr->subst_history.back().srcOps.size());
	}



	// we need delete the mapout_infor from curr node now
	//if (candidates[curr_node.father - del_counter].first_child == candidates[curr_node.father - del_counter].last_child) {
	if (candidates[curr_node.father - del_counter].further_search) {
		delete_mapout_infor_from_father(candidates[curr_node.father - del_counter], curr_node);
	}

}


//---------------------THIS FUNCTION ONLY FOR REUSE (WITH PARTIAL ORDER)----------------------//
// this function only find substs in affected range, and there should be a parameter telling which src op maps which new op
//add two new parameters here: 
//depthid: the smallest depth (i.e., srcop in the source graph of this xfer) which mapps a new op generated by last step
//newopid: which new op, i.e., the id in the dst ops of last step, is mapped
//if depthid and newopid == -1, then there is nothing to reuse, therefore no affected range here 
void GraphXfer::run_reuse(int depth, Graph* graph,
	std::deque<SearchTreeNode>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType,
	std::vector<std::vector<std::set<Op, OpCompare>>>& affected_range,
	size_t& add_counter, int depthid, int newopid, int curr_node_pos, double& substtime)
{
	//printf("searching globally\n");
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {

		//we need to check whether this subtitution would be redundant because of the symmetric rules
		/*if (success_for_symmetric_check(srcOps, substType) == false)
			return;*/
		if (found_symmetric_subst(graph, srcOps, substType))
			return;

		//printf("find all nodes to map\n");

		//check whether this partial sequence is redundant or not
		if (check(graph) == true)
			return; //may need to delete hash_map

		// Create dst operators
		bool pass = create_dst_ops_update_op_dict(substType, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		int count = 0;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
				Op& temOp = dstOp->mapOp;
				temOp.SIStepOrder = graph->subst_history.size();
				temOp.SIidx = count++;
			}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}


		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_order(graph, substType);

		// store cost change information of this step
		newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		///////////////////////////
		//newGraph->print_costs();
		///////////////////////////

		//if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
		//if (hashmap.find(newGraph->hash()) == hashmap.end()) {
		//printf("find one candidate!   %zu\n", candidates.size());
		//hashmap.insert(newGraph->hash());

		//candidates.push_back(newGraph);
		SearchTreeNode new_searchTreeNode(newGraph, curr_node_pos);
		candidates.push_back(new_searchTreeNode);
		//store meda infor for later substitution
		store_infor(candidates.back());

		//add edge in the search tree
		//std::vector<std::vector<size_t>>& brothers = candidates.at(curr_graph_pos).childrenNode;
		//brothers[substType].push_back(add_counter);

		add_counter = add_counter + 1;

		//print the new graph infor
		//print_subst_history_n_cost_subst(newGraph);
		
		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;
		
	}
	else {
		//printf("substType: %d     depth: %d    max depth:%zu   depthid: %d\n", substType, depth, srcOps.size(), depthid);
		if (depthid == -1) {
			//printf("depthid = -1\n");
			//iterate over all ops in the graph
			OpX* srcOp = srcOps[depth];
			std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
			for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
				if (can_match(srcOp, it->first, graph)
					&& (mappedOps.find(it->first) == mappedOps.end())) {
					Op op = it->first;
					// Check mapOutput
					match(srcOp, op, graph);
					run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos, substtime);
					unmatch(srcOp, op, graph);
				}
			}
		}
		else {
			//printf("depthid != -1\n");
			if (depth == depthid) {
				//the only option of mapped op here is corresponding to newopid
				OpX* srcOp = srcOps[depth];
				Op& possible_op = graph->subst_history.back().dstOps[newopid];
				if (can_match(srcOp, possible_op, graph)
					&& (mappedOps.find(possible_op) == mappedOps.end())) {
					//Op op = it->first;
					// Check mapOutput
					match(srcOp, possible_op, graph);
					run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos, substtime);
					unmatch(srcOp, possible_op, graph);
				}
			}
			else {
				OpX* srcOp = srcOps[depth];
				//try every possible op in the corresponding affected range
				//first we need to know the shortest distance from this srcop (depth) to depthid
				int dis = dis_matrix[depth][depthid];
				if (dis >= 0) {
					//printf("dis >= 0 ----------------- IS CONNECTED\n");
					//if the two src nodes are connected
					//then we iterate over the affected range of this distance of the mapped op (newopid)
					std::set<Op, OpCompare>::const_iterator it;
					const std::set<Op, OpCompare>& all_options = affected_range[newopid][dis];
					//std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
					for (it = all_options.begin(); it != all_options.end(); it++) {
						//printf("dis >= 0 depth %d\n", depth);
						if (depth < depthid) {
							//this op cannot map any new op from last step
							//printf("SIStepOrder: %d    size-1: %d\n", it->SIStepOrder, (int)(graph->subst_history.size() - 1));
							if (it->SIStepOrder == (int)(graph->subst_history.size() - 1))
								continue;
						}
						//printf("222\n");
						if (can_match(srcOp, *it, graph)
							&& (mappedOps.find(*it) == mappedOps.end())) {
							//printf("333\n");
							Op op = *it;
							// Check mapOutput
							match(srcOp, op, graph);
							//printf("mapped input size: %zu\n", mappedInputs.size());
							/*if (it->ptr == NULL)
							printf("this op is unwrapped!!!!!\n");*/
							/*printf("the number of inputs of the mapped op: %zu, %d\n", graph->inEdges.at(*it).size(), it->ptr->numInputs);
							printf("how many inputs of the srcop: %zu\n", srcOp->inputs.size());
							printf("THE INPUTS OF THIS SRCOP ARE \n");
							for (size_t haha = 0; haha < srcOp->inputs.size(); haha++) {
							if (srcOp->inputs[haha].op == NULL)
							printf("%d ", srcOp->inputs[haha].idx);
							}
							printf("\n");*/


							//printf("444\n");
							run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos, substtime);
							//printf("mapped input size: %zu\n", mappedInputs.size());
							//printf("555\n");
							unmatch(srcOp, op, graph);
							//printf("666\n");
						}
					}
					//printf("end for-----------\n");
				}
				else {
					//printf("dis < 0 ----------------- Not CONNECTED\n");
					//the two src nodes are not connected
					std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
					for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
						//printf("dis < 0\n");
						//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
						if (depth < depthid) {
							//this op cannot map any new op from last step
							if (it->first.SIStepOrder == (int)(graph->subst_history.size() - 1))
								continue;
						}

						if (can_match(srcOp, it->first, graph)
							&& (mappedOps.find(it->first) == mappedOps.end())) {
							Op op = it->first;
							// Check mapOutput
							match(srcOp, op, graph);
							run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos, substtime);
							unmatch(srcOp, op, graph);
						}
					}
				}
			}

		}
	}
}



//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION (WITHOUT PARTIAL ORDER)----------------------//
// this function only find substs in affected range, and there should be a parameter telling which src op maps which new op
//add two new parameters here: 
//depthid: the smallest depth (i.e., srcop in the source graph of this xfer) which mapps a new op generated by last step
//newopid: which new op, i.e., the id in the dst ops of last step, is mapped
//if depthid and newopid == -1, then there is nothing to reuse, therefore no affected range here 
//THE AFFECTED RANGE HERE IS SPECIFIED TO THE NEWOP_ID
//THE selected_map_op HERE IS THE OP CORR. TO NEWOP_ID
void GraphXfer::run_reuse(int depth, Graph* graph,
	std::deque<SearchTreeNode>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substType,
	std::vector<std::set<Op, OpCompare>>& affected_range,
	Op selected_map_op, const std::set<Op, OpCompare>& all_selected_ops,
	size_t& add_counter, int depthid, int curr_node_pos, double& substtime)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {

		//we need to check whether this subtitution would be redundant because of the symmetric rules
		/*if (success_for_symmetric_check(srcOps, substType) == false)
			return;*/
		if (found_symmetric_subst(graph, srcOps, substType))
			return;

		//printf("find all nodes to map\n");

		//check whether this partial sequence is redundant or not
		//if (check(graph) == true)
		//	return; //may need to delete hash_map

		// Create dst operators
		bool pass = create_dst_ops_update_op_dict(substType, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		int count = 0;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
				Op& temOp = dstOp->mapOp;
				temOp.SIStepOrder = graph->subst_history.size();
				temOp.SIidx = count++;
			}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}


		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_order(graph, substType);

		// store cost change information of this step
		newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		///////////////////////////
		//newGraph->print_costs();
		///////////////////////////

		//if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
		//if (hashmap.find(newGraph->hash()) == hashmap.end()) {
		//printf("find one candidate!   %zu\n", candidates.size());
		//hashmap.insert(newGraph->hash());


		//candidates.push_back(newGraph);
		SearchTreeNode new_searchTreeNode(newGraph, curr_node_pos);
		candidates.push_back(new_searchTreeNode);
		//store meda infor for later substitution
		store_infor(candidates.back());

		//add edge in the search tree
		//std::vector<std::vector<size_t>>& brothers = candidates.at(curr_graph_pos).childrenNode;
		//brothers[substType].push_back(add_counter);

		add_counter = add_counter + 1;

		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

	}
	else {
		
		/*if (substType == 150)
			printf("substType: %d     depth: %d    max depth:%zu   depthid: %d\n", substType, depth, srcOps.size(), depthid);*/

		if (depthid == -1) {
			//printf("depthid = -1\n");
			//iterate over all ops in the graph
			OpX* srcOp = srcOps[depth];
			std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
			for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
				if (can_match(srcOp, it->first, graph)
					&& (mappedOps.find(it->first) == mappedOps.end())) {
					Op op = it->first;
					// Check mapOutput
					match(srcOp, op, graph);
					run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, substtime);
					unmatch(srcOp, op, graph);
				}
			}
		}
		else {
			//printf("depthid != -1\n");
			if (depth == depthid) {
				//the only option of mapped op here is corresponding to newopid
				OpX* srcOp = srcOps[depth];
				//Op& possible_op = graph->subst_history.back().dstOps[newopid];
				Op& possible_op = selected_map_op;
				if (can_match(srcOp, possible_op, graph)
					&& (mappedOps.find(possible_op) == mappedOps.end())) {
					//Op op = it->first;
					// Check mapOutput
					match(srcOp, possible_op, graph);
					run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, substtime);
					unmatch(srcOp, possible_op, graph);
				}
			}
			else {
				OpX* srcOp = srcOps[depth];
				//try every possible op in the corresponding affected range
				//first we need to know the shortest distance from this srcop (depth) to depthid
				int dis = dis_matrix[depth][depthid];
				if (dis >= 0) {
					//printf("dis >= 0 ----------------- IS CONNECTED\n");
					//if the two src nodes are connected
					//then we iterate over the affected range of this distance of the mapped op (newopid)
					std::set<Op, OpCompare>::const_iterator it;
					//const std::set<Op, OpCompare>& all_options = affected_range[newopid][dis];
					const std::set<Op, OpCompare>& all_options = affected_range[dis];
					//std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
					for (it = all_options.begin(); it != all_options.end(); it++) {
						
						/*if (substType == 150)
							printf("dis >= 0 depth %d  guid %zu\n", depth, it->guid);*/
						
						
						if (depth < depthid) {
							//this op cannot map any new op from last step
							//if (it->SIStepOrder == (int)(graph->subst_history.size() - 1))
							if (all_selected_ops.find(*it)!= all_selected_ops.end())
								continue;
						}
						
						/*if (substType == 150)
							printf("222\n");*/
						
						
						if (can_match(srcOp, *it, graph)
							&& (mappedOps.find(*it) == mappedOps.end())) {
							//printf("333\n");
							Op op = *it;
							// Check mapOutput
							match(srcOp, op, graph);
							//printf("mapped input size: %zu\n", mappedInputs.size());
							/*if (it->ptr == NULL)
							printf("this op is unwrapped!!!!!\n");*/
							/*printf("the number of inputs of the mapped op: %zu, %d\n", graph->inEdges.at(*it).size(), it->ptr->numInputs);
							printf("how many inputs of the srcop: %zu\n", srcOp->inputs.size());
							printf("THE INPUTS OF THIS SRCOP ARE \n");
							for (size_t haha = 0; haha < srcOp->inputs.size(); haha++) {
							if (srcOp->inputs[haha].op == NULL)
							printf("%d ", srcOp->inputs[haha].idx);
							}
							printf("\n");*/

							/*if (substType == 150)
								printf("444\n");*/

							run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, substtime);
							//run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos);
							//printf("mapped input size: %zu\n", mappedInputs.size());
							//printf("555\n");
							unmatch(srcOp, op, graph);
							//printf("666\n");
						}
					}
					//printf("end for-----------\n");
				}
				else {
					//printf("dis < 0 ----------------- Not CONNECTED\n");
					//the two src nodes are not connected
					std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
					for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
						//printf("dis < 0\n");
						//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
						if (depth < depthid) {
							//this op cannot map any new op from last step
							//if (it->first.SIStepOrder == (int)(graph->subst_history.size() - 1))
							if (all_selected_ops.find(it->first) != all_selected_ops.end())
								continue;
						}

						if (can_match(srcOp, it->first, graph)
							&& (mappedOps.find(it->first) == mappedOps.end())) {
							Op op = it->first;
							// Check mapOutput
							match(srcOp, op, graph);
							run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, substtime);
							//run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos);
							unmatch(srcOp, op, graph);
						}
					}
				}
			}

		}
	}
}

//---------------------THIS FUNCTION ONLY FOR SAMPLETRICK LOCAL VERSION (WITHOUT PARTIAL ORDER)----------------------//
// this function only find substs in affected range, and there should be a parameter telling which src op maps which new op
//add two new parameters here: 
//depthid: the smallest depth (i.e., srcop in the source graph of this xfer) which mapps a new op generated by last step
//newopid: which new op, i.e., the id in the dst ops of last step, is mapped
//if depthid and newopid == -1, then there is nothing to reuse, therefore no affected range here 
//THE AFFECTED RANGE HERE IS SPECIFIED TO THE NEWOP_ID
//THE selected_map_op HERE IS THE OP CORR. TO NEWOP_ID
//THIS FUNCTION WOULD NOT STORE THE GRAPH FOUND, IT JUST FIND THE BEST SUBSTITUTION BASED ON THE ORIGINAL GRAPH
void GraphXfer::get_potential(int depth, Graph* graph,
	std::deque<SearchTreeNode>& candidates,
	int substType,
	std::vector<std::set<Op, OpCompare>>& affected_range,
	Op selected_map_op, const std::set<Op, OpCompare>& all_selected_ops,
	size_t& add_counter, int depthid, int curr_node_pos, float& potential)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {

		//we need to check whether this subtitution would be redundant because of the symmetric rules
		/*if (success_for_symmetric_check(srcOps, substType) == false)
			return;*/
		if (found_symmetric_subst(graph, srcOps, substType))
			return;

		//printf("find all nodes to map\n");

		//check whether this partial sequence is redundant or not
		//if (check(graph) == true)
		//	return; //may need to delete hash_map

		// Create dst operators
		bool pass = create_dst_ops_update_op_dict(substType, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		int count = 0;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
		if (pass) {
		OpX* dstOp = *dstIt;
		pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
		Op& temOp = dstOp->mapOp;
		temOp.SIStepOrder = graph->subst_history.size();
		temOp.SIidx = count++;
		}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}



		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_order(graph, substType);

		// store cost change information of this step
		//newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();
		

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;
			return;
		}

		float cost_change = 0;
		for (size_t i = 0; i < srcOps.size(); i++)
		{
			cost_change = cost_change - srcOps[i]->mapOp.ptr->runtime;
		}
		for (size_t i = 0; i < dstOps.size(); i++)
		{
			cost_change = cost_change + dstOps[i]->mapOp.ptr->runtime;
		}
		if (cost_change < potential)
			potential = cost_change;
		
		delete newGraph;
	}
	else {
		//printf("substType: %d     depth: %d    max depth:%zu   depthid: %d\n", substType, depth, srcOps.size(), depthid);
		if (depthid == -1) {
			//printf("depthid = -1\n");
			//iterate over all ops in the graph
			OpX* srcOp = srcOps[depth];
			std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
			for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
				if (can_match(srcOp, it->first, graph)
					&& (mappedOps.find(it->first) == mappedOps.end())) {
					Op op = it->first;
					// Check mapOutput
					match(srcOp, op, graph);
					get_potential(depth + 1, graph, candidates, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, potential);
					unmatch(srcOp, op, graph);
				}
			}
		}
		else {
			//printf("depthid != -1\n");
			if (depth == depthid) {
				//the only option of mapped op here is corresponding to newopid
				OpX* srcOp = srcOps[depth];
				//Op& possible_op = graph->subst_history.back().dstOps[newopid];
				Op& possible_op = selected_map_op;
				if (can_match(srcOp, possible_op, graph)
					&& (mappedOps.find(possible_op) == mappedOps.end())) {
					//Op op = it->first;
					// Check mapOutput
					match(srcOp, possible_op, graph);
					get_potential(depth + 1, graph, candidates, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, potential);
					unmatch(srcOp, possible_op, graph);
				}
			}
			else {
				OpX* srcOp = srcOps[depth];
				//try every possible op in the corresponding affected range
				//first we need to know the shortest distance from this srcop (depth) to depthid
				int dis = dis_matrix[depth][depthid];
				if (dis >= 0) {
					//printf("dis >= 0 ----------------- IS CONNECTED\n");
					//if the two src nodes are connected
					//then we iterate over the affected range of this distance of the mapped op (newopid)
					std::set<Op, OpCompare>::const_iterator it;
					//const std::set<Op, OpCompare>& all_options = affected_range[newopid][dis];
					const std::set<Op, OpCompare>& all_options = affected_range[dis];
					//std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
					for (it = all_options.begin(); it != all_options.end(); it++) {
						//printf("dis >= 0 depth %d\n", depth);
						if (depth < depthid) {
							//this op cannot map any new op from last step
							//if (it->SIStepOrder == (int)(graph->subst_history.size() - 1))
							if (all_selected_ops.find(*it) != all_selected_ops.end())
								continue;
						}
						//printf("222\n");
						if (can_match(srcOp, *it, graph)
							&& (mappedOps.find(*it) == mappedOps.end())) {
							//printf("333\n");
							Op op = *it;
							// Check mapOutput
							match(srcOp, op, graph);
							//printf("mapped input size: %zu\n", mappedInputs.size());
							/*if (it->ptr == NULL)
							printf("this op is unwrapped!!!!!\n");*/
							/*printf("the number of inputs of the mapped op: %zu, %d\n", graph->inEdges.at(*it).size(), it->ptr->numInputs);
							printf("how many inputs of the srcop: %zu\n", srcOp->inputs.size());
							printf("THE INPUTS OF THIS SRCOP ARE \n");
							for (size_t haha = 0; haha < srcOp->inputs.size(); haha++) {
							if (srcOp->inputs[haha].op == NULL)
							printf("%d ", srcOp->inputs[haha].idx);
							}
							printf("\n");*/


							//printf("444\n");
							get_potential(depth + 1, graph, candidates, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, potential);
							//run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos);
							//printf("mapped input size: %zu\n", mappedInputs.size());
							//printf("555\n");
							unmatch(srcOp, op, graph);
							//printf("666\n");
						}
					}
					//printf("end for-----------\n");
				}
				else {
					//printf("dis < 0 ----------------- Not CONNECTED\n");
					//the two src nodes are not connected
					std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
					for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
						//printf("dis < 0\n");
						//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
						if (depth < depthid) {
							//this op cannot map any new op from last step
							//if (it->first.SIStepOrder == (int)(graph->subst_history.size() - 1))
							if (all_selected_ops.find(it->first) != all_selected_ops.end())
								continue;
						}

						if (can_match(srcOp, it->first, graph)
							&& (mappedOps.find(it->first) == mappedOps.end())) {
							Op op = it->first;
							// Check mapOutput
							match(srcOp, op, graph);
							get_potential(depth + 1, graph, candidates, substType, affected_range, selected_map_op, all_selected_ops, add_counter, depthid, curr_node_pos, potential);
							//run_reuse(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substType, affected_range, add_counter, depthid, newopid, curr_node_pos);
							unmatch(srcOp, op, graph);
						}
					}
				}
			}

		}
	}
}

//---------------------THIS FUNCTION ONLY FOR REUSE----------------------//
void GraphXfer::store_infor(SearchTreeNode& search_tree_node) {
	/*search_tree_node.meta_mappedOps = mappedOps;
	search_tree_node.meta_mappedInputs = mappedInputs;
	search_tree_node.meta_mappedOutputs = mappedOutputs;
	search_tree_node.meta_dstOps = dstOps;*/

	//to save memory, we need to store the last step in the search tree node
	//search_tree_node.subst = search_tree_node.graphPtr->subst_history.back();

	//we store the output tensor mapping function here	
	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>& mapout_infor = search_tree_node.map_output_infor;
	std::map<TensorX, TensorX, TensorXCompare>::const_iterator it;
	for (it = mappedOutputs.begin(); it != mappedOutputs.end(); it++) {
		Op& old_op = it->first.op->mapOp;
		int old_idx = it->first.idx;
		Op& new_op = it->second.op->mapOp;
		int new_idx = it->second.idx;

		Edge e(old_op, new_op, old_idx, new_idx);
		mapout_infor[old_op].insert(e);
	}
}



//---------------------THIS FUNCTION FOR PARTIAL ORDER----------------------//
//check whether a partial sequence is redundant or not
bool GraphXfer::check(Graph* graph)
{
	//printf("check called!\n");
	if (graph->subst_history.size() == 0) {
		//because of the new partial order, we need to find the biggest node of the first step
		biggestNode = mappedOps.begin()->first;
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const Op& mappednode = opIt->first;
			//Graph::GraphSubst* opStep = mappednode.SIstep;
			if (compNode(biggestNode, mappednode) < 0)
				biggestNode = mappednode;
		}
		//printf("find no dependency!\n");

		//printf("print the biggest node of this step: StepOrder: %d    IDX: %d\n", biggestNode.SIStepOrder, biggestNode.SIidx);
		return false;
	}
	if (compStep(graph->subst_history.back()) < 0)
		return false;
	else
		return true;
}

//---------------------THIS FUNCTION FOR PARTIAL ORDER----------------------//
//compare two optimization steps
//input: last optimization step
int GraphXfer::compStep(Graph::GraphSubst& last)//, int substType)
{
	//if (last == NULL) // if last subst is None
	//{
	//	//if (substType == -1) // if this subst is also None
	//	//	return 0; // they are equal
	//	//else
	//		return -1; //last < this
	//}
	//else
	//{
	//if (substType == -1)
	//	return 1; //last > this
	//else
	//{
	//check whether this depends on last
	/*printf("compStep called!\n");*/
	//std::map<Op, OpX*, OpCompare>::const_iterator opIt;
	//for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
	//	const Op& mappednode = opIt->first;
	//	//Graph::GraphSubst* opStep = mappednode.SIstep;
	//	if (mappednode.SIStepOrder == last.order)
	//		return -1;
	//}

	//find the biggest node in the mappedOps
	biggestNode = mappedOps.begin()->first;
	std::map<Op, OpX*, OpCompare>::const_iterator opIt;
	for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
		const Op& mappednode = opIt->first;
		//Graph::GraphSubst* opStep = mappednode.SIstep;
		if (compNode(biggestNode, mappednode) < 0)
			biggestNode = mappednode;
	}

	//printf("find no dependency!\n");

	//printf("print the biggest node of this step: StepOrder: %d    IDX: %d\n", biggestNode.SIStepOrder, biggestNode.SIidx);


	//in fact, we do not need to compare the substType of two subst steps. 
	/*if (substType > last.substType)
	return -1;
	if (substType < last.substType)
	return 1;*/
	//return compNode(last.srcOps[0]);
	return compNode(last.biggestNode, biggestNode);
	//}
	//}
}



//---------------------THIS FUNCTION FOR PARTIAL ORDER----------------------//
//compare two nodes in the computation graph
//input: two nodes to be compared
//output: if first node < second node, return -1; if >, return 1; else, return 0.
int GraphXfer::compNode(const Op& firstNode, const Op& secondNode)
{
	if (firstNode.SIStepOrder < secondNode.SIStepOrder)
		return -1;
	if (firstNode.SIStepOrder > secondNode.SIStepOrder)
		return 1;
	int firstIdx = firstNode.SIidx;
	int secondIdx = secondNode.SIidx;
	if (firstIdx == -1)
		firstIdx = firstNode.guid;
	if (secondIdx == -1)
		secondIdx = secondNode.guid;
	//printf("lastIdx:%d           thisIdx:%d!\n", lastIdx, thisIdx);
	if (firstIdx < secondIdx)
		return -1;
	if (firstIdx > secondIdx)
		return 1;
	else
		return 0;
}


//---------------------THIS FUNCTION FOR PARTIAL ORDER----------------------//
Graph* GraphXfer::create_new_graph_order(Graph* graph, int substType)
{
	Graph* newGraph = new Graph();
	newGraph->subst_history = graph->subst_history;
	Graph::GraphSubst subst;
	for (size_t i = 0; i < srcOps.size(); i++) {
		Op op = srcOps[i]->mapOp;
		subst.srcOps.push_back(op);
	}
	for (size_t i = 0; i < dstOps.size(); i++) {
		Op op = dstOps[i]->mapOp;
		subst.dstOps.push_back(op);
	}
	subst.substType = substType;
	//subst.guid = ;
	subst.order = graph->subst_history.size();
	subst.biggestNode = biggestNode;
	newGraph->subst_history.push_back(subst);

	// Step 1: map dst ops
	std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
	std::vector<OpX*>::const_iterator dstIt;
	// Step 2: add edges to the graph
	for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
		if (mappedOps.find(opIt->first) == mappedOps.end()) {
			// Unmapped ops
			const std::set<Edge, EdgeCompare>& list = opIt->second;
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->srcOp) != mappedOps.end()) {
					// mapped src -> unmapped dst
					TensorX srcTen;
					srcTen.op = mappedOps[it->srcOp];
					srcTen.idx = it->srcIdx;
					assert(mappedOutputs.find(srcTen) != mappedOutputs.end());
					TensorX dstTen = mappedOutputs[srcTen];
					newGraph->add_edge(dstTen.op->mapOp, it->dstOp, dstTen.idx, it->dstIdx);
				}
				else {
					// unmapped src -> unmmaped dst
					newGraph->add_edge(it->srcOp, it->dstOp, it->srcIdx, it->dstIdx);
				}
		}
	// Step 3: add edges for mapped ops
	for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++) {
		OpX* dstOp = *dstIt;
		for (size_t i = 0; i < dstOp->inputs.size(); i++)
			if (dstOp->inputs[i].op == NULL) {
				// unmapped src -> mapped dst
				std::multimap<int, std::pair<Op, int> >::const_iterator it
					= mappedInputs.find(dstOp->inputs[i].idx);
				assert(it != mappedInputs.end());
				std::pair<Op, int> srcEdge = it->second;
				newGraph->add_edge(srcEdge.first, dstOp->mapOp, srcEdge.second, i);
			}
			else {
				// mapped src -> mapped dst
				OpX* srcOp = dstOp->inputs[i].op;
				int srcIdx = dstOp->inputs[i].idx;
				newGraph->add_edge(srcOp->mapOp, dstOp->mapOp, srcIdx, i);
			}
	}
	return newGraph;
}




//---------------------THIS FUNCTION ONLY FOR SYSMLPARTITION----------------------//
// par_i denotes which partition this function is now processing
void GraphXfer::run_partition(int depth, Graph* graph,
	std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substtype, int par_i, double& substtime)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {
		// Create dst operators
		bool pass = create_dst_ops_update_op_dict_SYS(substtype, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
			}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_sysmlpartition(graph);

		// store cost change information of this step
		newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();
		// store the subst type
		newGraph->subst_history.back().substType = substtype;

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		//print subst history
		//printf("        ===== Print subst history =====\n\n");
		//for (size_t i = 0; i < newGraph->subst_history.size(); i++) {
		//Graph::GraphSubst subst_print = newGraph->subst_history[i];
		////printf("        substitution[%03zu]  order:%d  type:%d   :\n", i, subst.order, subst.substType);
		//printf("        substitution[%03zu]  :\n", i);
		//for (size_t j = 0; j < subst_print.srcOps.size(); j++) {
		//printf("            srcOp[%zu]: %s   \n", j, subst_print.srcOps[j].to_string().c_str());
		//}
		//for (size_t j = 0; j < subst_print.dstOps.size(); j++) {
		//printf("            dstOp[%zu]: %s   \n", j, subst_print.dstOps[j].to_string().c_str());
		//}
		//}
		//printf("        latest type:%d   \n", substtype);

		//print successful subst
		//printf("        ===== Applied Substitutions =====\n\n");
		/*for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: \n", i);*/
		/*Graph::GraphSubst subst = newGraph->subst_history.back();
		printf("            substType %d\n", substtype);
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		newGraph->print_costs();*/
		//}

		//update partition infor
		//we now need to delete the partition infor of old edges and add that of new edges, and we assume that we never map a wrapped input or weight op 
		std::map<Edge, int, EdgeCompare>::const_iterator parinfor_it;
		for (parinfor_it = graph->par_infor.begin(); parinfor_it != graph->par_infor.end(); parinfor_it++) {
			if ((mappedOps.find(parinfor_it->first.srcOp) == mappedOps.end()) && (mappedOps.find(parinfor_it->first.dstOp) == mappedOps.end())) {
				//this edge will remain in the new graph
				newGraph->par_infor[parinfor_it->first] = parinfor_it->second; //this edge remain in its original partition
			}
		}
		Graph::GraphSubst& subst = newGraph->subst_history.back();
		for (size_t i = 0; i < subst.dstOps.size(); i++) {
			const Op& new_op = subst.dstOps[i];
			//check new in edge
			const std::set<Edge, EdgeCompare>& new_in_list = newGraph->inEdges[new_op];
			std::set<Edge, EdgeCompare>::const_iterator new_in_it;
			for (new_in_it = new_in_list.begin(); new_in_it != new_in_list.end(); new_in_it++) {
				newGraph->par_infor[*new_in_it] = par_i;
				//assert that we never map wrapped input or weight op
				assert(new_in_it->srcOp.ptr != NULL);
			}
			
			//check new out edge
			const std::set<Edge, EdgeCompare>& new_out_list = newGraph->outEdges[new_op];
			std::set<Edge, EdgeCompare>::const_iterator new_out_it;
			for (new_out_it = new_out_list.begin(); new_out_it != new_out_list.end(); new_out_it++) {
				newGraph->par_infor[*new_out_it] = par_i;
			}
		}
		
		//newGraph->par_infor = graph->par_infor;
		//Graph::GraphSubst& subst = newGraph->subst_history.back();
		//for (size_t i = 0; i < subst.srcOps.size(); i++) {
		//	//which_par = newGraph->par_infor[subst.srcOps[i]];
		//	newGraph->par_infor.erase(subst.srcOps[i]);
		//}
		//for (size_t i = 0; i < subst.dstOps.size(); i++) {
		//	newGraph->par_infor[subst.dstOps[i]] = par_i;
		//}

		if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
			//if (newGraph->total_cost() < threshold) {
			if (hashmap.find(newGraph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(newGraph->hash());
				candidates.push(newGraph);
			}
			else
				delete newGraph;
		}
		else {
			delete newGraph;
		}

		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

	}
	else {
		OpX* srcOp = srcOps[depth];
		std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
		//std::set<Op, OpCompare>::const_iterator it;
		//for (it = ops_in_par.begin(); it != ops_in_par.end(); it++) {
		for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
			//if (graph->par_infor[it->first] != par_i) // only match op in this partition
			//	continue;

			//only match edge in this partition			
			const std::set<Edge, EdgeCompare>& in_list = it->second; //all in edges of this op
			std::set<Edge, EdgeCompare>::const_iterator in_it;
			bool check_edge = true;
			for (in_it = in_list.begin(); in_it != in_list.end(); in_it++) {
				if (in_it->srcOp.ptr != NULL) {
					//this srcOp is a valid op and must has been partitioned
					assert(graph->par_infor.find(*in_it) != graph->par_infor.end());
					if (graph->par_infor[*in_it] != par_i) {
						check_edge = false;
						break;
					}
				}
				//if ((graph->par_infor.find(in_it->srcOp) != graph->par_infor.end()) && (graph->par_infor[in_it->srcOp] != par_i)) {
				//	//the in edge is not in this partition
				//	check_edge = false;
				//	break;
				//}
			}
			if (!check_edge)
				continue;
			const std::set<Edge, EdgeCompare>& out_list = graph->outEdges[it->first]; //all out edges of this op, if this op is not in the list, insert it.
			std::set<Edge, EdgeCompare>::const_iterator out_it;
			for (out_it = out_list.begin(); out_it != out_list.end(); out_it++) {
				//this edge must has been partitioned
				if (graph->par_infor[*out_it] != par_i) {
					//the out edge is not in this partition
					check_edge = false;
					break;
				}
				//if (graph->par_infor[out_it->dstOp] != par_i) {
				//	//the in edge is not in this partition
				//	check_edge = false;
				//	break;
				//}
			}
			if (!check_edge)
				continue;
			
			//now we are sure that all inedges and outedges of this op is in this partition
			if (can_match(srcOp, it->first, graph)
				&& (mappedOps.find(it->first) == mappedOps.end())) {
				Op op = it->first;
				// Check mapOutput
				match(srcOp, op, graph);
				run_partition(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substtype, par_i, substtime);
				unmatch(srcOp, op, graph);
			}
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR SYSMLPARTITION----------------------//
void GraphXfer::run_boundary(int depth, Graph* graph,
	std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substtype, double& substtime)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {

		// we need to first check whether this substitution spans the partition boundary or not
		bool cross_bound = false;
		int which_par = -1; //store the first par whose edges this substitution needs
		std::map<Op, OpX*, OpCompare>::const_iterator cross_It;
		for (cross_It = mappedOps.begin(); cross_It != mappedOps.end(); cross_It++) {
			const std::set<Edge, EdgeCompare>& in_list = graph->inEdges[cross_It->first]; //all in edges of this op
			std::set<Edge, EdgeCompare>::const_iterator in_it;
			for (in_it = in_list.begin(); in_it != in_list.end(); in_it++) {
				assert(graph->par_infor.find(*in_it) != graph->par_infor.end());
				if (graph->par_infor[*in_it] == -2) {
					//this edge is generated during local search
					cross_bound = true;
					break;
				}
				else {
					if (which_par == -1) {
						which_par = graph->par_infor[*in_it]; //this is the first partition this subst step covers
					}
					else {
						if (which_par != graph->par_infor[*in_it]) {
							//this substitution crosses partitions
							cross_bound = true;
							break;
						}
					}
				}
				
				//if ((graph->par_infor.find(in_it->srcOp) != graph->par_infor.end()) && (graph->par_infor.find(in_it->dstOp) != graph->par_infor.end()) 
				//	&& (graph->par_infor[in_it->srcOp] != graph->par_infor[in_it->dstOp])) {
				//	// contain one crossing-boundary edge
				//	cross_bound = true;
				//	break;
				//}
			}
			if (cross_bound)
				break;
			const std::set<Edge, EdgeCompare>& out_list = graph->outEdges[cross_It->first]; //all out edges of this op, if this op is not in the list, insert it.
			std::set<Edge, EdgeCompare>::const_iterator out_it;
			for (out_it = out_list.begin(); out_it != out_list.end(); out_it++) {
				assert(graph->par_infor.find(*out_it) != graph->par_infor.end());
				if (graph->par_infor[*out_it] == -2) {
					//this edge is generated during local search
					cross_bound = true;
					break;
				}
				else {
					if (which_par == -1) {
						which_par = graph->par_infor[*out_it];
					}
					else {
						if (which_par != graph->par_infor[*out_it]) {
							//this substitution crosses partitions
							cross_bound = true;
							break;
						}
					}
				}

				//if (mappedOps.find(out_it->dstOp) == mappedOps.end()) {
				//	// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
				//	if ((graph->par_infor.find(out_it->srcOp) != graph->par_infor.end()) && (graph->par_infor.find(out_it->dstOp) != graph->par_infor.end())
				//		&& (graph->par_infor[out_it->srcOp] != graph->par_infor[out_it->dstOp])) {
				//		// contain one crossing-boundary edge
				//		cross_bound = true;
				//		break;
				//	}
				//}
			}
			if (cross_bound)
				break;
		}

		if (!cross_bound) //if does not cross boundary
			return;

		// Create dst operators
		bool pass = create_dst_ops_update_op_dict_SYS(substtype, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
			}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_sysmlpartition(graph);

		// store cost change information of this step
		newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();
		// store the subst type
		newGraph->subst_history.back().substType = substtype;

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		//print successful subst
		//printf("        ===== Applied Substitutions =====\n\n");
		/*for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: \n", i);*/
		/*Graph::GraphSubst subst = newGraph->subst_history.back();
		printf("            substType %d\n", substtype);
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		newGraph->print_costs();*/
		//}

		//newGraph->par_infor = graph->par_infor;

		//update partition infor
		//we now need to delete the partition infor of old edges and add that of new edges, and we assume that we never map a wrapped input or weight op 
		std::map<Edge, int, EdgeCompare>::const_iterator parinfor_it;
		for (parinfor_it = graph->par_infor.begin(); parinfor_it != graph->par_infor.end(); parinfor_it++) {
			if ((mappedOps.find(parinfor_it->first.srcOp) == mappedOps.end()) && (mappedOps.find(parinfor_it->first.dstOp) == mappedOps.end())) {
				//this edge will remain in the new graph
				newGraph->par_infor[parinfor_it->first] = parinfor_it->second; //this edge remain in its original partition
			}
		}
		Graph::GraphSubst& subst = newGraph->subst_history.back();
		for (size_t i = 0; i < subst.dstOps.size(); i++) {
			const Op& new_op = subst.dstOps[i];
			//check new in edge
			const std::set<Edge, EdgeCompare>& new_in_list = newGraph->inEdges[new_op];
			std::set<Edge, EdgeCompare>::const_iterator new_in_it;
			for (new_in_it = new_in_list.begin(); new_in_it != new_in_list.end(); new_in_it++) {
				newGraph->par_infor[*new_in_it] = -2;
				//assert that we never map wrapped input or weight op
				assert(new_in_it->srcOp.ptr != NULL);
			}

			//check new out edge
			const std::set<Edge, EdgeCompare>& new_out_list = newGraph->outEdges[new_op];
			std::set<Edge, EdgeCompare>::const_iterator new_out_it;
			for (new_out_it = new_out_list.begin(); new_out_it != new_out_list.end(); new_out_it++) {
				newGraph->par_infor[*new_out_it] = -2;
			}
		}




		//print subst history
		//printf("        ===== Print subst history =====\n\n");
		//for (size_t i = 0; i < newGraph->subst_history.size(); i++) {
		//	Graph::GraphSubst subst_print = newGraph->subst_history[i];
		//	//printf("        substitution[%03zu]  order:%d  type:%d   :\n", i, subst.order, subst.substType);
		//	printf("        substitution[%03zu]  :\n", i);
		//	for (size_t j = 0; j < subst_print.srcOps.size(); j++) {
		//		printf("            srcOp[%zu]: %s   \n", j, subst_print.srcOps[j].to_string().c_str());
		//	}
		//	for (size_t j = 0; j < subst_print.dstOps.size(); j++) {
		//		printf("            dstOp[%zu]: %s   \n", j, subst_print.dstOps[j].to_string().c_str());
		//	}
		//}
		//printf("        latest type:%d   \n", substtype);



		if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
			//if (newGraph->total_cost() < threshold) {
			if (hashmap.find(newGraph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(newGraph->hash());
				candidates.push(newGraph);
			}
			else
				delete newGraph;
		}
		else {
			delete newGraph;
		}

		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

	}
	else {
		OpX* srcOp = srcOps[depth];
		std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
		for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
			//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
			if (can_match(srcOp, it->first, graph)
				&& (mappedOps.find(it->first) == mappedOps.end())) {
				Op op = it->first;
				// Check mapOutput
				match(srcOp, op, graph);
				run_boundary(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substtype, substtime);
				unmatch(srcOp, op, graph);
			}
		}
	}
}


//---------------------THIS FUNCTION ONLY FOR SYSMLTRICK WITHOUT PARTITION----------------------//
void GraphXfer::run_sysmltrick(int depth, Graph* graph,
	std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substtype, double& substtime)
	//void GraphXfer::run(int depth, Graph* graph,
	//	std::deque<Graph*>& candidates,
	//	std::set<size_t>& hashmap, float threshold, int maxNumOps, int substtype)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {
		// Create dst operators
		bool pass = create_dst_ops_update_op_dict_SYS(substtype, graph);
		/*bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
			}*/
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}

		//RECORD THE START OF SUBST
		auto start_time = std::chrono::system_clock::now();

		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_sysmlpartition(graph);

		// store cost change information of this step
		newGraph->subst_history.back().cost_change = newGraph->total_cost() - graph->total_cost();
		// store the subst type
		newGraph->subst_history.back().substType = substtype;

		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;

			//RECORD THE END OF SUBST
			auto end_time = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
			double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
			substtime += searchtime;

			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		//print successful subst
		//printf("        ===== Applied Substitutions =====\n\n");
		/*for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		printf("        substitution[%03zu]: \n", i);*/
		/*Graph::GraphSubst subst = newGraph->subst_history.back();
		printf("            substType %d\n", substtype);
		for (size_t j = 0; j < subst.srcOps.size(); j++) {
		printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		}
		for (size_t j = 0; j < subst.dstOps.size(); j++) {
		printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		}
		newGraph->print_costs();*/
		//}

		if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
			//if (newGraph->total_cost() < threshold) {
			if (hashmap.find(newGraph->hash()) == hashmap.end()) {
				//printf("add candidate!\n");
				hashmap.insert(newGraph->hash());
				candidates.push(newGraph);
			}
			else
				delete newGraph;
		}
		else {
			delete newGraph;
		}

		//RECORD THE END OF SUBST
		auto end_time = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		double searchtime = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
		substtime += searchtime;

	}
	else {
		OpX* srcOp = srcOps[depth];
		std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
		for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
			//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
			if (can_match(srcOp, it->first, graph)
				&& (mappedOps.find(it->first) == mappedOps.end())) {
				Op op = it->first;
				// Check mapOutput
				match(srcOp, op, graph);
				run_sysmltrick(depth + 1, graph, candidates, hashmap, threshold, maxNumOps, substtype, substtime);
				unmatch(srcOp, op, graph);
			}
		}
	}
}



//---------------------THIS FUNCTION ONLY FOR SYSMLPARTITION----------------------//
//This function collects the vertex weight for graph partitioning, which is stored in vertex_weight
void GraphXfer::collect_vertex_weight(int depth, Graph* graph, 
	std::map<Op, int, OpCompare>& vertex_weight, int substtype)
{
	//printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
	if (depth >= (int)srcOps.size()) {
		// Create dst operators
		bool pass = true;
		std::vector<OpX*>::const_iterator dstIt;
		for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
			if (pass) {
				OpX* dstOp = *dstIt;
				pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
			}
		if (!pass) return;
		// Check that output tensors with external edges are mapped
		std::map<Op, OpX*, OpCompare>::const_iterator opIt;
		for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
			const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
			std::set<Edge, EdgeCompare>::const_iterator it;
			for (it = list.begin(); it != list.end(); it++)
				if (mappedOps.find(it->dstOp) == mappedOps.end()) {
					// dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
					TensorX srcTen;
					srcTen.op = opIt->second;
					srcTen.idx = it->srcIdx;
					if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
						pass = false;
						return;
					}
				}
		}
		// Generate a new graph by applying xfer rule
		Graph* newGraph = create_new_graph_sysmlpartition(graph);
		// Check that the new graph should not have any loop
		if (newGraph->has_loop()) {
			//printf("Found a new graph with LOOP!!!!\n");
			delete newGraph;
			return;
		}
		// TODO: remove me for better performance
		assert(newGraph->check_correctness());

		//print successful subst
		//printf("        ===== Applied Substitutions =====\n\n");
		///*for (size_t i = 0; i < bestGraph->subst_history.size(); i++) {
		//printf("        substitution[%03zu]: \n", i);*/
		//Graph::GraphSubst subst = newGraph->subst_history.back();
		//printf("            substType %d\n", substtype);
		//for (size_t j = 0; j < subst.srcOps.size(); j++) {
		//printf("            srcOp[%zu]: %s\n", j, subst.srcOps[j].to_string().c_str());
		//}
		//for (size_t j = 0; j < subst.dstOps.size(); j++) {
		//printf("            dstOp[%zu]: %s\n", j, subst.dstOps[j].to_string().c_str());
		//}
		//newGraph->print_costs();
		//}

		//Now we can assure that the substitution can be applied successfully
		//Therefore, we need to update the vertex weight according to this substitution
		std::map<Op, OpX*, OpCompare>::const_iterator opIt_for_vw;
		for (opIt_for_vw = mappedOps.begin(); opIt_for_vw != mappedOps.end(); opIt_for_vw++) {
			//we need to check this mapped op has both input edges and output edges
			if ( (graph->inEdges[opIt_for_vw->first].size() > 0) && (graph->outEdges[opIt_for_vw->first].size() > 0) ) {
				if (vertex_weight.find(opIt_for_vw->first) == vertex_weight.end())
					vertex_weight[opIt_for_vw->first] = 1;
				else
					vertex_weight[opIt_for_vw->first] ++;
			}
		}	

		//since we only collect vertex weight, we delete the newGraph
		delete newGraph;
	}
	else {
		OpX* srcOp = srcOps[depth];
		std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
		for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
			//printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
			if (can_match(srcOp, it->first, graph)
				&& (mappedOps.find(it->first) == mappedOps.end())) {
				Op op = it->first;
				// Check mapOutput
				match(srcOp, op, graph);
				collect_vertex_weight(depth + 1, graph, vertex_weight, substtype);
				unmatch(srcOp, op, graph);
			}
		}
	}
}

//---------------------THIS FUNCTION FOR SYSMLPARTITION and SYSMLTRICK (WITHOUT PARTITION)----------------------//
Graph* GraphXfer::create_new_graph_sysmlpartition(Graph* graph)
{
  Graph* newGraph = new Graph();
  newGraph->subst_history = graph->subst_history;
  Graph::GraphSubst subst;
  for (size_t i = 0; i < srcOps.size(); i++) {
    Op op = srcOps[i]->mapOp;
    subst.srcOps.push_back(op);
  }
  for (size_t i = 0; i < dstOps.size(); i++) {
    Op op = dstOps[i]->mapOp;
    subst.dstOps.push_back(op);
  }
  newGraph->subst_history.push_back(subst);

  // Step 1: map dst ops
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  std::vector<OpX*>::const_iterator dstIt;
  // Step 2: add edges to the graph
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mappedOps.find(opIt->first) == mappedOps.end()) {
      // Unmapped ops
      const std::set<Edge, EdgeCompare>& list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mappedOps.find(it->srcOp) != mappedOps.end()) {
          // mapped src -> unmapped dst
          TensorX srcTen;
          srcTen.op = mappedOps[it->srcOp];
          srcTen.idx = it->srcIdx;
          assert(mappedOutputs.find(srcTen) != mappedOutputs.end());
          TensorX dstTen = mappedOutputs[srcTen];
          newGraph->add_edge(dstTen.op->mapOp, it->dstOp, dstTen.idx, it->dstIdx);
        } else {
          // unmapped src -> unmmaped dst
          newGraph->add_edge(it->srcOp, it->dstOp, it->srcIdx, it->dstIdx);
        }
    }
  // Step 3: add edges for mapped ops
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt ++) {
    OpX* dstOp = *dstIt;
    for (size_t i = 0; i < dstOp->inputs.size(); i++)
      if (dstOp->inputs[i].op == NULL) {
        // unmapped src -> mapped dst
        std::multimap<int, std::pair<Op, int> >::const_iterator it
            = mappedInputs.find(dstOp->inputs[i].idx);
        assert(it != mappedInputs.end());
        std::pair<Op, int> srcEdge = it->second;
        newGraph->add_edge(srcEdge.first, dstOp->mapOp, srcEdge.second, i);
      } else {
        // mapped src -> mapped dst
        OpX* srcOp = dstOp->inputs[i].op;
        int srcIdx = dstOp->inputs[i].idx;
        newGraph->add_edge(srcOp->mapOp, dstOp->mapOp, srcIdx, i);
      }
  }
  return newGraph;
}


bool GraphXfer::create_new_operator(const OpX* opx, Op& op)
{
  switch (opx->type) {
    case OP_CONV2D:
    {
      assert(opx->inputs.size() == 2);
      Tensor input = opx->inputs[0].to_tensor(this);
      Tensor weight = opx->inputs[1].to_tensor(this);
      int strideH, strideW, padding, activation;
      assert(opx->get_pm_constraint(PM_STRIDE_H, strideH));
      assert(opx->get_pm_constraint(PM_STRIDE_W, strideW));
      assert(opx->get_pm_constraint(PM_PAD, padding));
      assert(opx->get_pm_constraint(PM_ACTI, activation));
      op = model->get_or_create_conv2d(input, weight, strideH, strideW,
                                       (PaddingMode)padding,
                                       (ActiMode)activation);
      break;
    }
    case OP_EW_ADD:
    case OP_EW_MUL:
    {
      assert(opx->inputs.size() == 2);
      Tensor input0 = opx->inputs[0].to_tensor(this);
      Tensor input1 = opx->inputs[1].to_tensor(this);
      op = model->get_or_create_element(opx->type, input0, input1);
      break;
    }
    case OP_MATMUL:
    {
      assert(opx->inputs.size() == 2);
      Tensor input = opx->inputs[0].to_tensor(this);
      Tensor weight = opx->inputs[1].to_tensor(this);
      int activation;
      assert(opx->get_pm_constraint(PM_ACTI, activation));
      op = model->get_or_create_matmul(input, weight,
                                       (ActiMode)activation);
      break;
    }
    case OP_TRANSPOSE:
    {
      assert(opx->inputs.size() == 1);
      Tensor input = opx->inputs[0].to_tensor(this);
      int permIdx, shuffle;
      assert(opx->get_pm_constraint(PM_PERM, permIdx));
      assert(opx->get_pm_constraint(PM_OUTSHUFFLE, shuffle));
      op = model->get_or_create_transpose(input, permIdx, (bool)shuffle);
      break;
    }
    case OP_ENLARGE:
    {
      assert(opx->inputs.size() == 2);
      Tensor w1 = opx->inputs[0].to_tensor(this);
      Tensor w2 = opx->inputs[1].to_tensor(this);
      //int kernelH, kernelW;
      //assert(opx->get_pm_constraint(PM_KERNEL_H, kernelH));
      //assert(opx->get_pm_constraint(PM_KERNEL_W, kernelW));
      
	  //printf("!!!!ENLARGING!\n");
	  
	  op = model->get_or_create_enlarge(w1, w2);
      break;
    }
    case OP_MERGE_GCONV:
    {
      assert(opx->inputs.size() == 1);
      Tensor weight = opx->inputs[0].to_tensor(this);
      int count;
      assert(opx->get_pm_constraint(PM_MERGE_GCONV_COUNT, count));
      op = model->get_or_create_merge_gconv(weight, count);
      break;
    }
    case OP_CONCAT:
    {
      // TODO: assume don't need copy for now
      Tensor inputs[MAX_NUM_INPUTS];
      bool needCopy[MAX_NUM_INPUTS];
      for (size_t i = 0; i < opx->inputs.size(); i++) {
        inputs[i] = opx->inputs[i].to_tensor(this);
        needCopy[i] = false;
      }
      int axis;
      assert(opx->get_pm_constraint(PM_AXIS, axis));
      op = model->get_or_create_concat(axis, opx->inputs.size(), inputs, needCopy);
      break;
    }
    case OP_SPLIT:
    {
      int axis;
      Tensor input = opx->inputs[0].to_tensor(this);
      assert(opx->get_pm_constraint(PM_AXIS, axis));
      op = model->get_or_create_split(input, axis, opx->outputs.size());
      break;
    }
    case OP_RELU:
    case OP_TANH:
    case OP_SIGMOID:
    {
      assert(opx->inputs.size() == 1);
      Tensor input = opx->inputs[0].to_tensor(this);
      op = model->get_or_create_activation(input, opx->type, true);
      break;
    }
    default:
    {
      printf("opx->type = %d\n", opx->type);
      assert(false);
    }
  }
  // Check operator validness
  if (op == Op::INVALID_OP)
    return false;
  // Check tnConstraints
  for (size_t i = 0; i < opx->tnConstraints.size(); i++) {
    TNConstraint tnc = opx->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op.ptr->get_input_parameter(tnc.para2, tnc.dim2, &expValue));
    }
    switch (tnc.comp) {
      case COMPARE_EQ:
        if (actValue != expValue) return false;
        break;
      case COMPARE_NE:
        if (actValue == expValue) return false;
        break;
      case COMPARE_LT:
        if (actValue >= expValue) return false;
        break;
      case COMPARE_LE:
        if (actValue > expValue) return false;
        break;
      case COMPARE_GT:
        if (actValue <= expValue) return false;
        break;
      case COMPARE_GE:
        if (actValue < expValue) return false;
        break;
      default:
        assert(false);
    }
  }
  return true;
}

/*
void GraphXfer::run(int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::set<size_t>& hashmap, float threshold)
{
  if (depth >= srcOps.size()) {
    // Check two op constraints
    bool pass = true;
    for (size_t i = 0; i < constraints.size(); i++) {
      TwoOpConstraint toc = constraints[i];
      int value1, value2;
      assert(toc.op1->mapOp.ptr != NULL);
      assert(toc.op2->mapOp.ptr != NULL);
      assert(toc.op1->mapOp.ptr->get_parameter(toc.para1, &value1));
      assert(toc.op2->mapOp.ptr->get_parameter(toc.para2, &value2));
      switch (toc.comp) {
        case COMPARE_EQ:
          if (value1 != value2) pass = false;
          break;
        case COMPARE_NE:
          if (value1 == value2) pass = false;
          break;
        case COMPARE_LT:
          if (value1 >= value2) pass = false;
          break;
        case COMPARE_GT:
          if (value1 <= value2) pass = false;
          break;
        default:
          assert(false);
      }
    }
    // Generate a new graph by applying xfer rule
    if (pass) {
      Graph* newGraph = create_new_graph(graph);
      //assert(newGraph->check_correctness());
      if (newGraph->total_cost() < threshold) {
        if (hashmap.find(newGraph->hash()) == hashmap.end()) {
          hashmap.insert(newGraph->hash());
          candidates.push(newGraph);
        }
      } else {
        delete newGraph;
      }
    }
  } else {
    // Match srcOps[depth];
    SrcOp* srcOp = srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      if (srcOp->match(it->first)
      && (mapped.find(it->first) == mapped.end())) {
        Op op = it->first;
        std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> > list = srcInEdges[srcOp];
        std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> >::const_iterator it2;
        // Check edges in the source subgraph
        bool pass = true;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          SubEdge<SrcOp> edge = *it2;
          if (!graph->has_edge(edge.srcOp->mapOp, op, edge.srcIdx, edge.dstIdx)) pass = false;
        }
        // Check mapInput/mapOutput
        bool extraInputs = false, extraOutputs = false;
        if (srcInEdges[srcOp].size() != graph->num_in_edges(op))
          extraInputs = true;
        if (srcOutEdges[srcOp].size() != graph->num_out_edges(op))
          extraOutputs = true;
        if (!srcOp->mapInput && extraInputs)
          pass = false;
        if (!srcOp->mapOutput && extraOutputs)
          pass = false;
        // Serch for the next op if pass the check
        if (pass) {
          srcOp->mapOp = op;
          mapped.insert(op);
          run(depth + 1, graph, candidates, hashmap, threshold);
          mapped.erase(op);
          srcOp->mapOp.guid = 0;
          srcOp->mapOp.ptr = NULL;
        }
      }
    }
  }
}

Graph* GraphXfer::create_new_graph(Graph* graph)
{
  Graph* newGraph = new Graph(graph->model);
  // Step 1: add operators to the graph
  std::vector<DstOp*>::iterator dstIt;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mapped.find(opIt->first) == mapped.end()) {
      newGraph->inEdges[opIt->first];
      newGraph->outEdges[opIt->first];
    }
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt ++) {
    DstOp* dstOp = *dstIt;
    dstOp->mapOp = dstOp->create_operator(graph->model);
    newGraph->inEdges[dstOp->mapOp];
    newGraph->outEdges[dstOp->mapOp];
  }
  // Step 2: add edges to the graph
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mapped.find(opIt->first) != mapped.end()) {
      // Mapped ops
      std::set<Edge, EdgeCompare> list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mapped.find(it->srcOp) != mapped.end()) {
          // mapped src -> mapped dst
          // Do nothing!
        } else {
          // unmapped src -> mapped dst
          int i = 0;
          for (i = 0; i < srcOps.size(); i++)
            if (srcOps[i]->mapOp.guid == opIt->first.guid) break;
          assert(i < srcOps.size());
          assert(srcOps[i]->mapInput != NULL);
          Op op = srcOps[i]->mapInput->mapOp;
          Edge e(it->srcOp, op, it->srcIdx, it->dstIdx);
          newGraph->inEdges[op].insert(e);
          newGraph->outEdges[it->srcOp].insert(e);
        }
    } else {
      // Unmapped ops
      std::set<Edge, EdgeCompare> list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mapped.find(it->srcOp) != mapped.end()) {
          // mapped src -> unmapped dst
          int i = 0;
          for (i = 0; i < srcOps.size(); i++)
            if (srcOps[i]->mapOp.guid == it->srcOp.guid) break;
          assert(i < srcOps.size());
          assert(srcOps[i]->mapOutput != NULL);
          Op op = srcOps[i]->mapOutput->mapOp;
          Edge e(op, opIt->first, it->srcIdx, it->dstIdx);
          newGraph->inEdges[opIt->first].insert(e);
          newGraph->outEdges[op].insert(e);
        } else {
          // unmapped src -> unmapped dst
          Edge e(it->srcOp, opIt->first, it->srcIdx, it->dstIdx);
          newGraph->inEdges[opIt->first].insert(e);
          newGraph->outEdges[it->srcOp].insert(e);
        }
    }
  // Step 3: add edges in the dstInEdges
  std::map<DstOp*, std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > >::iterator dstOpIt;
  for (dstOpIt = dstInEdges.begin(); dstOpIt != dstInEdges.end(); dstOpIt++) {
    std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > list = dstOpIt->second;
    std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> >::const_iterator it;
    for (it = list.begin(); it != list.end(); it++) {
      Op src = it->srcOp->mapOp, dst = dstOpIt->first->mapOp;
      Edge e(src, dst, it->srcIdx, it->dstIdx);
      newGraph->inEdges[dst].insert(e);
      newGraph->outEdges[src].insert(e);
    }
  }
  return newGraph;
}
*/
