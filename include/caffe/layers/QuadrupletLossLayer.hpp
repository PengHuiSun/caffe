#ifndef CAFFE_LAYER_QUADRUPLET_LOSS_LAYER_HPP__
#define CAFFE_LAYER_QUADRUPLET_LOSS_LAYER_HPP__

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include <vector>

namespace caffe {

template <typename Dtype> struct QFunction {
  Dtype margin_;
  Dtype age_margin_;
  Dtype curvature_;

  void setup(const LayerParameter &param) {
    margin_ = param.qfunction_param().margin();
    age_margin_ =
        param.qfunction_param().age_margin(); // age margin for function Q
    curvature_ =
        param.qfunction_param().curvature(); // curvature for function Q
  }

  inline Dtype call(const Dtype &y1, const Dtype &y2) const {
    return Q(y1, y2);
  }

  inline Dtype Q_raw(const Dtype &y1, const Dtype &y2) const {
    Dtype abs_ = abs(y1 - y2);
    Dtype ret1 = age_margin_ * log(1 + abs_ / curvature_);
    if (abs_ <= age_margin_) {
      return ret1;
    } else {
      Dtype C = age_margin_ - ret1;
      return abs_ - C;
    }
  }

  inline Dtype Q(const Dtype &y1, const Dtype &y2) const {
    Dtype abs_ = abs(y1 - y2);
    Dtype ret1 = age_margin_ * log(1 + abs_ / curvature_);
    if (abs_ <= age_margin_) {
      return ret1 / Q_raw(Dtype(0), age_margin_);
    } else {
      Dtype C = age_margin_ - ret1;
      return (abs_ - C) / Q_raw(Dtype(0), Dtype(69)); // 0 min age, 69 max age
    }
  };
};

template <typename Dtype> class QuadrupletLossLayer : public LossLayer<Dtype> {
public:
  explicit QuadrupletLossLayer(const LayerParameter &param)
      : LossLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  virtual inline const char *type() const { return "SunPengHuiLoss"; }

  inline Dtype margin() const { return qFunc_.margin_; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> helper_;
  QFunction<Dtype> qFunc_;
};

} // namespace caffe

#endif // CAFFE_LAYER_QUADRUPLET_LOSS_LAYER_HPP__
