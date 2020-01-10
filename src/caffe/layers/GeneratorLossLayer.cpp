#include "caffe/layers/GeneratorLossLayer.hpp"

#include <cmath>

namespace caffe {

template <typename Dtype>
void GeneratorLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    CHECK_EQ(bottom[0]->height(), 1);
    CHECK_EQ(bottom[0]->width(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());

    diff_.ReshapeLike(*bottom[0]);
    helper_.Reshape(1, bottom[0]->channels(), 1, 1);
    margin_ = this->layer_param_.generator_loss_param().margin();
    age_margin_ = this->layer_param_.generator_loss_param().age_margin(); // age margin for function Q
    curvature_ = this->layer_param_.generator_loss_param().curvature();   // curvature for function Q
}

template <typename Dtype>
void GeneratorLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // bottom[0] original
    // bottom[1] generate
    // bottom[2] label

    int N = bottom[0]->num(); // get the batch size
    int S = bottom[0]->height() * bottom[0]->width() * bottom[0]->channels(); // get each blob child size
    const Dtype* origin = bottom[0]->cpu_data();
    const Dtype* generate = bottom[1]->cpu_data();
    const Dtype* label = bottom[2]->cpu_data();
    Dtype tmp0(0.0), tmp1(0.0), tmp2(0.0), qTmp(0.0), this_loss(0.0);
    Dtype loss(0.0);

    for (int i = 0; i < N; i += 4) {
        CHECK_EQ(label[i + 0], label[i + 1]);
        CHECK_NE(label[i + 0], label[i + 2]);
        CHECK_NE(label[i + 1], label[i + 3]);
        CHECK_NE(label[i + 2], label[i + 3]);
    }

    Dtype lambda = this->layer_param_.generator_loss_param().lambda();
    Dtype theta = this->layer_param_.generator_loss_param().theta();

    // caffe_set(diff_.count(), Dtype(0.0), diff_.mutable_cpu_data());

    for (int i = 0; i < N; i += 4) {
        const Dtype* A   = &origin[(i + 0) * S];
        const Dtype* P   = &origin[(i + 1) * S];
        const Dtype* N1  = &origin[(i + 2) * S];
        const Dtype* N2  = &origin[(i + 3) * S];
        const Dtype* N11 = &generate[(i + 2) * S];
        const Dtype* N22 = &generate[(i + 3) * S];

        Dtype* diff_A   = &diff_.mutable_cpu_data()[(i + 0) * S];
        Dtype* diff_P   = &diff_.mutable_cpu_data()[(i + 1) * S];
        Dtype* diff_N11 = &diff_.mutable_cpu_data()[(i + 2) * S];
        Dtype* diff_N22 = &diff_.mutable_cpu_data()[(i + 3) * S]; 

        // clear diff A
        caffe_set(S, Dtype(0.0), diff_A);

        // clear diff P
        caffe_set(S, Dtype(0.0), diff_P);
        
        // D(N11, A)
        caffe_sub(S, A, N11, diff_N11);
        tmp0 = caffe_cpu_dot(S, diff_N11, diff_N11);
        
        // D(N11, N1)
        caffe_sub(S, N1, N11, helper_.mutable_cpu_data());
        caffe_cpu_axpby(S, Dtype(1.0), helper_.cpu_data(), Dtype(1.0), diff_N11);
        tmp1 = caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());

        // D(N11, A) * (1 - Q)
        qTmp = 1 - Q(label[i + 0], label[i + 2]);
        tmp2 = tmp0 * qTmp - margin_;

        if (tmp2 <= Dtype(0.0)) {
            qTmp = Dtype(0.0);
        }
        caffe_cpu_axpby(S, qTmp * theta, diff_N11, Dtype(1.0), diff_N11);

        // G = D(N11, A) + D(N11, N1) + D(N11, A) * (1 - Q)
        this_loss = tmp0 + tmp1 + theta * std::max(Dtype(0), tmp2);

        loss += this_loss;

        // For (P, N2, N22)

        // D(N22, P)
        caffe_sub(S, P, N22, diff_N22);
        tmp0 = caffe_cpu_dot(S, diff_N22, diff_N22);
        
        // D(N22, N1)
        caffe_sub(S, N2, N22, helper_.mutable_cpu_data());
        caffe_cpu_axpby(S, Dtype(1.0), helper_.cpu_data(), Dtype(1.0), diff_N22);
        tmp1 = caffe_cpu_dot(S, helper_.cpu_data(), helper_.cpu_data());

        // D(N22, P) * (1 - Q)
        qTmp = 1 - Q(label[i + 1], label[i + 3]);
        tmp2 = tmp0 * qTmp - margin_;
        if (tmp2 <= Dtype(0.0)) {
            qTmp = Dtype(0.0);
        }
        caffe_cpu_axpby(S, qTmp * theta, diff_N22, Dtype(1.0), diff_N22);

        // G = D(N22, P) + D(N22, N2) + D(N22, A) * (1 - Q)
        this_loss = tmp0 + tmp1 + theta * std::max(Dtype(0), tmp2);

        loss += this_loss;
    }

    loss = loss / (N/4);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void GeneratorLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        int N = bottom[1]->num();
        Dtype* bout = bottom[1]->mutable_cpu_diff();
        const Dtype alpha = top[0]->cpu_diff()[0] / (N/4) / Dtype(2.0);
        caffe_cpu_axpby(bottom[0]->count(), alpha, diff_.cpu_data(), Dtype(0.0), bout);
    }
}

#ifdef CPU_ONLY
STUB_GPU(GeneratorLossLayer);
#endif

INSTANTIATE_CLASS(GeneratorLossLayer);
REGISTER_LAYER_CLASS(GeneratorLoss);

}
