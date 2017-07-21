#include "LittleTest.hh"
#include "Sum.hh"
#include "Identity.hh"
#include "Cholesky.hh"
#include <random>
#include "Points.hh"
#include <Eigen/Dense>

template <typename T>
void triggerNextSample(T& vecs) {
    for (auto& vec: vecs ) vec.nextSample(); 
}

int main() {

    constexpr int number_of_vecs{4};
    constexpr size_t vec_size{100};
    constexpr int number_of_experiments{5};
    std::vector<RandomNormalVector> vecs;
    std::vector<MatrixHolder> mats;
    vecs.resize(number_of_vecs);
    mats.resize(number_of_vecs);

    Eigen::MatrixXd mat(vec_size, vec_size);
    mat.setRandom();
    for (int i=0; i<number_of_vecs; ++i) {
        vecs[i].finalize(0., 1., vec_size); 
        mats[i].finalize(mat);
    }

    auto sum = Sum();
    for (int i=0; i<number_of_vecs; ++i) {
        auto& vec = vecs[i];
        auto& matrix = mats[i];
        matrix.transformations.at(0).inputs(vec.transformations.at(0));
        auto trasf_output = matrix.transformations.at(0).outputs;
        auto vec_output =vec.transformations.at(0).outputs;
        sum.add(trasf_output);
    }

    EuclideanNorm normer;
    normer.transformations.at(0).inputs(sum);

    for (int exp=0; exp<number_of_experiments; ++exp){
    std::cout <<"\n Sample " << exp << "\n" << normer.transformations.at(0)[0].x << std::endl;
    triggerNextSample(vecs);
    }

    return 0;
}
