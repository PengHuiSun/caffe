#ifndef CAFFE_QUAD_EXPAND_LAYER_HPP__
#define CAFFE_QUAD_EXPAND_LAYER_HPP__

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class QuadExpandLayer : public Layer<Dtype> {
public:
  QuadExpandLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuadExpand"; }
  virtual inline int         ExactNumBottomBlobs() const { return 1; }
  virtual inline int         ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& top);
};

}  // namespace caffe

#endif  // CAFFE_QUAD_EXPAND_LAYER_HPP__