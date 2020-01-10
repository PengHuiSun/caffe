#include "caffe/layers/QuadMergeLayer.hpp"

namespace caffe {

template <typename Dtype> 
void QuadMergeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->channels(), bottom[0]->channels());
    CHECK_EQ(bottom[0]->width(), 1);
    CHECK_EQ(bottom[0]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
}

template <typename Dtype>
void QuadMergeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape = bottom[0]->shape();
    top[0]->Reshape(top_shape);
    // top[0]->ShareData(*bottom[0]);
    // top[0]->ShareDiff(*bottom[0]);
}

template <typename Dtype>
void QuadMergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // bottom[0] original
    // bottom[1] generate
    int N = bottom[0]->num();
    int S = bottom[0]->channels();
    const Dtype* generate = bottom[1]->mutable_cpu_data();
    const Dtype* original = bottom[0]->mutable_cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for (int i = 0; i < N; i += 4) {
        const Dtype*   A  = &original[(i + 0) * S];
        const Dtype*   P  = &original[(i + 1) * S];
        const Dtype*  N1  = &generate[(i + 2) * S];
        const Dtype*  N2  = &generate[(i + 3) * S];
        Dtype* tA  = &top_data[(i + 0) * S];
        Dtype* tP  = &top_data[(i + 1) * S];
        Dtype* tN1 = &top_data[(i + 2) * S];
        Dtype* tN2 = &top_data[(i + 3) * S];

        caffe_copy(S, A, tA);
        caffe_copy(S, P, tP);
        caffe_copy(S, N1, tN1);
        caffe_copy(S, N2, tN2);
    }
}

template <typename Dtype>
void QuadMergeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    }
    if (propagate_down[1]) {
        caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), bottom[1]->mutable_cpu_diff());
    }
}

#ifdef CPU_ONLY
STUB_GPU(QuadMergeLayer);
#endif

INSTANTIATE_CLASS(QuadMergeLayer);
REGISTER_LAYER_CLASS(QuadMerge);

}