#include "caffe/layers/GeneratorLossLayer.hpp"

#include <cmath>

namespace caffe {

template <typename Dtype>
void GeneratorLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());

  helper_.Reshape(1, bottom[1]->channels(), 1, 1);

  qFunc_.setup(this->layer_param_);
}

template <typename Dtype>
void GeneratorLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  // bottom[0] original
  // bottom[1] generate
  // bottom[2] label

  int N = bottom[0]->num(); // get the batch size
  int S = bottom[0]->height() * bottom[0]->width() *
          bottom[0]->channels(); // get each blob child size
  const Dtype *origin = bottom[0]->cpu_data();
  const Dtype *generate = bottom[1]->cpu_data();
  const Dtype *label = bottom[2]->cpu_data();
  Dtype *bout = bottom[1]->mutable_cpu_diff();
  Dtype tmp(0.0), this_loss(0.0);
  Dtype loss(0.0);
  num_constraints = 0.0;

  Dtype margin = qFunc_.margin_;

  for (int i = 0, j = 0; i < N; i += 4, j += 2) {
    const Dtype *A = &origin[(i + 0) * S];
    const Dtype *P = &origin[i * S + 1];
    const Dtype *N1 = &origin[i * S + 2];
    const Dtype *N2 = &origin[i * S + 3];
    const Dtype *N11 = &generate[(j + 0) * S];
    const Dtype *N22 = &generate[(j + 1) * S];

    Dtype *diff_N11 = &bout[(j + 0) * S];
    Dtype *diff_N22 = &bout[(j + 1) * S];

    // for negative 1
    caffe_sub(S, N11, A, helper_.mutable_cpu_data());
    caffe_copy(S, helper_.cpu_data(), diff_N11);
    this_loss = caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());

    caffe_sub(S, N11, N1, helper_.mutable_cpu_data());
    caffe_add(S, diff_N11, helper_.cpu_data(), diff_N11);
    this_loss += caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());

    caffe_sub(S, A, N11, helper_.mutable_cpu_data());
    tmp = caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());
    tmp = tmp * (1 - qFunc_.call(label[i + 0], label[i + 2])) - margin;
    tmp = std::max(Dtype(0), tmp);
    if (tmp != 0) {
      caffe_add(S, diff_N11, helper_.cpu_data(), diff_N11);
    }
    this_loss += tmp;

    loss += this_loss;

    // for negative 2
    caffe_sub(S, N22, P, helper_.mutable_cpu_data());
    caffe_copy(S, helper_.cpu_data(), diff_N22);
    this_loss = caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());

    caffe_sub(S, N22, N2, helper_.mutable_cpu_data());
    caffe_add(S, diff_N22, helper_.cpu_data(), diff_N22);
    this_loss += caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());

    caffe_sub(S, P, N22, helper_.mutable_cpu_data());
    tmp = caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());
    tmp = tmp * (1 - qFunc_.call(label[i + 1], label[i + 3])) - margin;
    tmp = std::max(Dtype(0), tmp);
    if (tmp != 0) {
      caffe_add(S, diff_N22, helper_.cpu_data(), diff_N22);
    }
    this_loss += tmp;
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void GeneratorLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

  if (propagate_down[1]) {
      const Dtype alpha = 2 * top[0]->cpu_diff()[0];
      const int N = bottom[1]->num();
      const int S = bottom[1]->channels() * bottom[1]->width() * bottom[1]->height();
      caffe_cpu_scale(N * S, alpha, bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(GeneratorLossLayer);
#endif

INSTANTIATE_CLASS(GeneratorLossLayer);
REGISTER_LAYER_CLASS(GeneratorLoss);

} // namespace caffe
