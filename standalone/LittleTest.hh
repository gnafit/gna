#ifndef LITTLETEST_H
#define LITTLETEST_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "GNAObject.hh"


class RandomNormalVector: public GNAObject,
                          public Transformation<RandomNormalVector> 
{
    public:
        RandomNormalVector() {
            transformation_(this, "gauss_mc")
                .output("gauss_mc")
                .types([](RandomNormalVector *obj, Atypes, Rtypes rets){
                        rets[0] = DataType().points().shape(obj->m_sample_size);})
                .func(&RandomNormalVector::calcMC);
        
        m_gen.seed(reinterpret_cast<size_t>(&m_gen));
        };

        void finalize(double mean, double var, size_t size) {
            m_dist.param(std::normal_distribution<double>::param_type(mean, var));
            m_sample_size = size;
        }

        void nextSample() {
            t_["gauss_mc"].unfreeze();
            t_["gauss_mc"].taint();
        };

    private:
        void calcMC(Args , Rets rets) {
            for (size_t el=0; el < rets[0].type.size(); ++el) {
                rets[0].x(el) = m_dist(m_gen);
            };
        };

        std::mt19937_64 m_gen;
        std::normal_distribution<double> m_dist;
        size_t m_sample_size;
};


class MatrixHolder:public GNAObject,
                   public Transformation<MatrixHolder> 
{
    public:
        MatrixHolder() {
            transformation_(this, "mat_product")
                .input("source")
                .output("sink")
                .types(&MatrixHolder::checkTypes, Atypes::pass<0>)
                .func(&MatrixHolder::computeProduct);
        };

        void finalize(const Eigen::MatrixXd& outer_mat) {
            m_mat = outer_mat;
        }

        void dump() {
            std::cout << m_mat <<"\n " << std::endl;
        }

    private:
        void checkTypes(Atypes args, Rtypes rets) { 
            if (m_mat.rows() != m_mat.cols()) {
                throw rets.error(rets[0], "Matrix is not square");
            }
            if (static_cast<int>(args[0].size()) != m_mat.cols()) {
                throw rets.error(rets[0], "Matrix and input have different sizes");
            }
        };

        void computeProduct(Args args, Rets rets) {
            rets[0].x = m_mat * args[0].vec;
        }


        Eigen::MatrixXd m_mat;
                   
};

class EuclideanNorm: public GNAObject,
                     public Transformation<EuclideanNorm>
{
    public:
        EuclideanNorm() {
            transformation_(this, "euclidean_norm")
                .input("vec")
                .output("norm")
                .types([](EuclideanNorm* obj, Atypes args, Rtypes rets)
                        {rets[0] = DataType().points().shape(1);})
                .func([](EuclideanNorm* obj, Args args, Rets rets)
                        {rets[0].x = args[0].vec.norm();});
        };
};

#endif //LITTLETEST_H
