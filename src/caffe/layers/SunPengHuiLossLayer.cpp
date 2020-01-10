#include "caffe/layers/SunPengHuiLossLayer.hpp"


namespace caffe {

template <typename Dtype>
void SunPengHuiLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    CHECK_EQ(bottom[0]->height(), 1);
    CHECK_EQ(bottom[0]->width(), 1);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    diff_.ReshapeLike(*bottom[0]);
    helper_.Reshape(1, bottom[0]->channels(), 1, 1);
    
    margin_ = this->layer_param_.sun_peng_hui_loss_param().margin();
    age_margin_ = this->layer_param_.sun_peng_hui_loss_param().age_margin(); // age margin for function Q
    curvature_ = this->layer_param_.sun_peng_hui_loss_param().curvature();   // curvature for function Q

}

template <typename Dtype>
void SunPengHuiLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void SunPengHuiLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // bottom[0] input
    // bottom[1] label

    int N = bottom[0]->num(); // get the batch size
    int S = bottom[0]->height() * bottom[0]->width() * bottom[0]->channels(); // get each blob child size
    const Dtype* b_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype loss(0.0);
    Dtype tmp0(0.0);
    Dtype tmp1(0.0);
    Dtype tmp2(0.0);
    Dtype qTmp(0.0);

    for (int i = 0; i < bottom[0]->num(); i += 4) {
        CHECK_EQ(label[i + 0], label[i + 1]);
        CHECK_NE(label[i + 0], label[i + 2]);
        CHECK_NE(label[i + 1], label[i + 3]);
        CHECK_NE(label[i + 2], label[i + 3]);
    }

    Dtype lambda = this->layer_param_.sun_peng_hui_loss_param().lambda();
    Dtype theta = this->layer_param_.sun_peng_hui_loss_param().theta();

    for (int i = 0; i < N; i+=4) {
        const Dtype* _A  = &b_data[(i + 0) * S];
        const Dtype* _P  = &b_data[(i + 1) * S];
        const Dtype* _N1 = &b_data[(i + 2) * S];
        const Dtype* _N2 = &b_data[(i + 3) * S];
        Dtype* diff_A =  &diff_.mutable_cpu_data()[(i + 0) * S];
        Dtype* diff_P =  &diff_.mutable_cpu_data()[(i + 1) * S];
        Dtype* diff_N1 = &diff_.mutable_cpu_data()[(i + 2) * S];
        Dtype* diff_N2 = &diff_.mutable_cpu_data()[(i + 3) * S];

        // D(A, P)
        caffe_sub(S, _P, _A, diff_P);
        tmp0 = caffe_cpu_dot(S, diff_P, diff_P);
        tmp0 = std::max(Dtype(0.0), tmp0);
        tmp0 *= tmp0;
        
        LOG_IF(INFO, Caffe::root_solver()) << " D(A,  P): " << tmp0;

        if (tmp0 == Dtype(0.0)) {
            caffe_scal(S, -lambda, diff_P);            
            caffe_cpu_axpby(S, lambda, diff_P, Dtype(0.0), diff_A);
        } else {
            caffe_scal(S, lambda, diff_P);
            caffe_cpu_axpby(S, -lambda, diff_P, Dtype(0.0), diff_A);
        }

        // D(N1, A)
        caffe_sub(S, _N1, _A, diff_N1);
        tmp1 = caffe_cpu_dot(S, diff_N1, diff_N1);
        
        LOG_IF(INFO, Caffe::root_solver()) << " D(N1, A): " << tmp1;

        // margin - D(N1, A) * Q
        qTmp = Q(label[i+0], label[i+2]);
        tmp1 = margin_ - tmp1 * qTmp;
        tmp1 = std::max(tmp1, Dtype(0.0));
        tmp1 *= tmp1;
        if (tmp1 == Dtype(0.0))
            qTmp = Dtype(0.0);
        caffe_scal(S, theta * qTmp, diff_N1);
        caffe_cpu_axpby(S, Dtype(-1.0), diff_N1, Dtype(1.0), diff_A);

        // D(N2, P)
        caffe_sub(S, _N2, _P, diff_N2);
        tmp2 = caffe_cpu_dot(S, diff_N2, diff_N2);

        LOG_IF(INFO, Caffe::root_solver()) << "D(N2, P): " << tmp2;

        // margin - D(N2, P) * Q
        qTmp = Q(label[i+1], label[i+3]);
        tmp2 = margin_ - tmp2 * qTmp;
        tmp2 = std::max(tmp1, Dtype(0.0));
        tmp2 *= tmp2;
        if (tmp2 == Dtype(0.0))
            qTmp = Dtype(0.0);
        caffe_scal(S, theta * qTmp, diff_N2);
        caffe_cpu_axpby(S, Dtype(-1.0), diff_N1, Dtype(1.0), diff_P);

        // loss += tmp0 + theta * tmp1 + theta * tmp2;
        loss += tmp0 + tmp1 + tmp2;
    }

    loss = loss / (N/4);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SunPengHuiLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0]) {
        int N = bottom[0]->num();
        const Dtype alpha = top[0]->cpu_diff()[0] / (N/4) * Dtype(2.0);
        Dtype* bout = bottom[0]->mutable_cpu_diff();

        caffe_cpu_axpby(bottom[0]->count(), alpha, diff_.cpu_data(), Dtype(0.0), bout);
    }
}

#ifdef CPU_ONLY
STUB_GPU(SunPengHuiLossLayer);
#endif

INSTANTIATE_CLASS(SunPengHuiLossLayer);
REGISTER_LAYER_CLASS(SunPengHuiLoss);

}