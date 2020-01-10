#ifndef CAFFE_LAYER_SUN_PENG_HUI_LOSS_LAYER_HPP__
#define CAFFE_LAYER_SUN_PENG_HUI_LOSS_LAYER_HPP__

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include <vector>

namespace caffe {

template <typename Dtype>
class SunPengHuiLossLayer : public LossLayer<Dtype> {
public:
  explicit SunPengHuiLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  virtual inline const char* type() const { return "SunPengHuiLoss"; }

  inline Dtype Q_raw(const Dtype& y1, const Dtype& y2) {
    Dtype abs_ = abs(y1 - y2);
    Dtype ret1 = age_margin_ * log(1 + abs_ / curvature_);
    if (abs_ <= age_margin_) {
      return ret1;
    } else {
      Dtype C = age_margin_ - ret1;
      return abs_ - C;
    }
  }

  inline Dtype Q(const Dtype& y1, const Dtype& y2) {
    Dtype abs_ = abs(y1 - y2);
    Dtype ret1 = age_margin_ * log(1 + abs_ / curvature_);
    if (abs_ <= age_margin_) {
      return ret1 / Q_raw(Dtype(0), age_margin_);
    } else {
      Dtype C = age_margin_ - ret1;
      return (abs_ - C) / Q_raw(Dtype(0), Dtype(69));  // 0 min age, 69 max age
    }
  };

  inline Dtype margin() const { return margin_; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);

  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype       margin_;
  Dtype       age_margin_;
  Dtype       curvature_;
  Blob<Dtype> diff_;
  Blob<Dtype> helper_;
};

}  // namespace caffe

#endif  // CAFFE_LAYER_SUN_PENG_HUI_LOSS_LAYER_HPP__