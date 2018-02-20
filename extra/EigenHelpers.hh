#pragma once

#include <iostream>
#include <Eigen/Dense>

namespace EigenHelpers{
    template <typename Derived>
        void dumpt(const Eigen::EigenBase<Derived>& mat) { std::cout<<mat<<std::endl; }

    void dump(const Eigen::MatrixXd& mat) { std::cout<<mat<<std::endl; }
    void dump(const Eigen::VectorXd& mat) { std::cout<<mat<<std::endl; }
    void dump(const Eigen::ArrayXd& mat)  { std::cout<<mat<<std::endl; }
    void dump(const Eigen::ArrayXXd& mat) { std::cout<<mat<<std::endl; }
}
