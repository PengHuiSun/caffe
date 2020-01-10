#include "caffe/layers/QuadExpandLayer.hpp"

namespace caffe {

template <typename Dtype>
void QuadExpandLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    CHECK_EQ(bottom[0]->width(), 1);
    CHECK_EQ(bottom[0]->height(), 1);
    CHECK_EQ(bottom[0]->num() % 4, 0);
}

template <typename Dtype>
void QuadExpandLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {    
    // vector<int> top_shape = bottom[0]->shape();
    // top_shape[2] *= 3;

    top[0]->Reshape(
        bottom[0]->num(),
        bottom[0]->channels() * 3,
        bottom[0]->height(),
        bottom[0]->width()
    );

    // check
    CHECK_EQ(bottom[0]->num(), top[0]->num());
    CHECK_EQ(bottom[0]->channels()*3, top[0]->channels());
    CHECK_EQ(bottom[0]->width(), top[0]->width());
    CHECK_EQ(bottom[0]->height(), top[0]->height());
}

template <typename Dtype>
void QuadExpandLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int N = bottom[0]->num();
    int S = bottom[0]->channels() * bottom[0]->width() * bottom[0]->height();

    int tK = top[0]->channels();
    int tS = top[0]->channels() * top[0]->width() * top[0]->height();

    // A refere to anchor, P refere to positive, N1 refere to negative 1, N2 refere to negative 2
    // the bottom input must be:
    // <A, P, N1, N2> <A, P, N1, N2> ... <A, P, N1, N2>
    // the top output is:
    // <AAA, PPP, N1AP, N2AP> <AAA, PPP, N1AP, N2AP> ... <AAA, PPP, N1AP, N2AP>

    for (int i = 0; i < N; i += 4) {
        // target A
        caffe_copy(S, &bottom_data[(i + 0) * S], &top_data[(i + 0) * tS + tK * 0]);
        caffe_copy(S, &bottom_data[(i + 0) * S], &top_data[(i + 0) * tS + tK * 1]);
        caffe_copy(S, &bottom_data[(i + 0) * S], &top_data[(i + 0) * tS + tK * 2]);

        // target P
        caffe_copy(S, &bottom_data[(i + 1) * S], &top_data[(i + 1) * tS + tK * 0]);
        caffe_copy(S, &bottom_data[(i + 1) * S], &top_data[(i + 1) * tS + tK * 1]);
        caffe_copy(S, &bottom_data[(i + 1) * S], &top_data[(i + 1) * tS + tK * 2]);

        // target N1
        caffe_copy(S, &bottom_data[(i + 2) * S], &top_data[(i + 2) * tS + tK * 0]);
        caffe_copy(S, &bottom_data[(i + 0) * S], &top_data[(i + 2) * tS + tK * 1]);
        caffe_copy(S, &bottom_data[(i + 1) * S], &top_data[(i + 2) * tS + tK * 2]);

        // target N2
        caffe_copy(S, &bottom_data[(i + 3) * S], &top_data[(i + 3) * tS + tK * 0]);
        caffe_copy(S, &bottom_data[(i + 0) * S], &top_data[(i + 3) * tS + tK * 1]);
        caffe_copy(S, &bottom_data[(i + 1) * S], &top_data[(i + 3) * tS + tK * 2]);
    }
}

template <typename Dtype>
void QuadExpandLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    caffe_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(QuadExpandLayer);
#endif

INSTANTIATE_CLASS(QuadExpandLayer);
REGISTER_LAYER_CLASS(QuadExpand);

}