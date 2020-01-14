#include "caffe/layers/QuadrupletLossLayer.hpp"

namespace caffe {

template <typename Dtype>
void QuadrupletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  int N = bottom[0]->num();

  diff_.Reshape(N / 4 * 3, bottom[0]->channels(), bottom[0]->height(),
                bottom[0]->width());
  helper_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                  bottom[0]->width());

  qFunc_.setup(this->layer_param_);
}

template <typename Dtype>
void QuadrupletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  // bottom[0] input
  // bottom[1] label

  int N = bottom[0]->num(); // get the batch size
  int S = bottom[0]->height() * bottom[0]->width() *
          bottom[0]->channels(); // get each blob child size
  const Dtype *b_data = bottom[0]->cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  Dtype loss(0.0);
  Dtype loss_a_p(0.0);
  Dtype loss_p_n2(0.0);
  Dtype loss_a_n1(0.0);
  Dtype q(0.0);

  for (int i = 0; i < bottom[0]->num(); i += 4) {
    CHECK_EQ(label[i + 0], label[i + 1]);
    CHECK_NE(label[i + 0], label[i + 2]);
    CHECK_NE(label[i + 1], label[i + 3]);
    CHECK_NE(label[i + 2], label[i + 3]);
  }

  for (int i = 0; i < N; i += 4) {
    const Dtype *_A = &b_data[(i + 0) * S];
    const Dtype *_P = &b_data[(i + 1) * S];
    const Dtype *_N1 = &b_data[(i + 2) * S];
    const Dtype *_N2 = &b_data[(i + 3) * S];
    Dtype *diff_A_N1 = &diff_.mutable_cpu_data()[(i + 0) * S];
    Dtype *diff_P_N2 = &diff_.mutable_cpu_data()[(i + 1) * S];
    Dtype *diff_A_P = &diff_.mutable_cpu_data()[(i + 2) * S];

    caffe_sub(S, _A, _N1, diff_A_N1);
    caffe_sub(S, _P, _N2, diff_P_N2);
    caffe_sub(S, _A, _P, diff_A_P);

    // compute: q of A N1
    q = qFunc_.call(label[i + 0], label[i + 2]);
    // compute: loss of A N1
    loss_a_n1 = margin - caffe_cpu_dot(S, diff_A_N1, diff_A_N1) * q;

    // for backward
    if (loss_a_n1 == 0) {
      caffe_set(S, Dtype(0), diff_A_N1);
    } else {
      caffe_cpu_scale(S, q, diff_A_N1, diff_A_N1);
    }

    // compute: q of P N2
    q = qFunc_.call(label[i + 1], lable[i + 4]);
    // compute: loss of P N2
    loss_p_n2 = margin - caffe_cpu_dot(S, diff_P_N2, diff_P_N2) * q;

    // for backward
    if (loss_a_n1 == 0) {
      caffe_set(S, Dtype(0), diff_P_N2);
    } else {
      caffe_cpu_scale(S, q, diff_P_N2, diff_P_N2);
    }

    // compute: loss of A P
    loss_a_p = caffe_cpu_dot(S, diff_A_P, diff_A_P);

    // for backward
    if (loss_a_n1 == 0) {
      caffe_set(S, Dtype(0), diff_A_P);
    } else {
      caffe_cpu_scale(S, q, diff_A_P, diff_A_P);
    }

    loss += loss_a_n1 + loss_p_n2 + loss_a_p
  }

  loss = loss / N;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void QuadrupletLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    int N = bottom[0]->num();
    int S = bottom[0]->height() * bottom[0]->width() *
            bottom[0]->channels(); // get each blob child size
    Dtype *b_data = bottom[0]->mutable_cpu_diff();
    const Dtype *diff = diff_.cpu_data();
    Dtype *helper = helper_.mutable_cpu_data();
    for (int i = 0; i < N; i += 4) {
      Dtype *_A = &b_data[(i + 0) * S];
      Dtype *_P = &b_data[(i + 1) * S];
      Dtype *_N1 = &b_data[(i + 2) * S];
      Dtype *_N2 = &b_data[(i + 3) * S];
      const Dtype *diff_A_N1 = diff[(i + 0) * S];
      const Dtype *diff_P_N2 = diff[(i + 1) * S];
      const Dtype *diff_A_P = diff[(i + 2) * S];

      Dtype alpha = 2 * top[0]->cpu_diff()[0];

      caffe_sub(S, diff_A_P, diff_A_N1, helper);
      caffe_cpu_scale(S, alpha, helper, _A);

      caffe_sub(S, diff_A_P, diff_P_N2, helper);
      caffe_cpu_scale(S, -alpha, helper, _P);

      caffe_cpu_scale(S, alpha, diff_A_N1, _N1);

      caffe_cpu_scale(S, alpha, diff_P_N2, _N2);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(QuadrupletLossLayer);
#endif

INSTANTIATE_CLASS(QuadrupletLossLayer);
REGISTER_LAYER_CLASS(QuadrupletLoss);

} // namespace caffe