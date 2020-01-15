#ifndef CAFFE_LAYERS_GENERATOR_LOSS_LAYER_HPP__
#define CAFFE_LAYERS_GENERATOR_LOSS_LAYER_HPP__

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/QuadrupletLossLayer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include <vector>


namespace caffe {

template <typename Dtype> class GeneratorLossLayer : public LossLayer<Dtype> {
public:
  GeneratorLossLayer(const LayerParameter &param)
      : LossLayer<Dtype>(param), helper_() {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top) {
    LossLayer<Dtype>::Reshape(bottom, top);
  }

  virtual inline int ExactNumBottomBlobs() const { return 3; }

  virtual inline const char *type() const { return "GeneratorLoss"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

  QFunction<Dtype> qFunc_;
  Blob<Dtype> helper_; // to help caculations
  Dtype num_constraints;
};

} // namespace caffe

#endif // CAFFE_LAYERS_GENERATOR_LOSS_LAYER_HPP__