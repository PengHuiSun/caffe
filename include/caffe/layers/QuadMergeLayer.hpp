#ifndef CAFFE_LAYERS_QUAD_MERGE_LAYER_HPP__
#define CAFFE_LAYERS_QUAD_MERGE_LAYER_HPP__

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class QuadMergeLayer : public Layer<Dtype> {
public:
  QuadMergeLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuadMerge"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& top);

  Blob<Dtype> helper_;
};

}  // namespace caffe

#endif  // CAFFE_LAYERS_QUAD_MERGE_LAYER_HPP__
