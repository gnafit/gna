#ifndef EIGENHELPERS_H
#define EIGENHELPERS_H 1

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace EigenHelpers{
    template <typename Derived>
        void dumpt(const Eigen::EigenBase<Derived>& mat) { std::cout<<mat<<std::endl; }

    inline void dump(const Eigen::MatrixXd& mat) { std::cout<<mat<<std::endl; }
    inline void dump(const Eigen::VectorXd& mat) { std::cout<<mat<<std::endl; }
    inline void dump(const Eigen::ArrayXd& mat)  { std::cout<<mat<<std::endl; }
    inline void dump(const Eigen::ArrayXXd& mat) { std::cout<<mat<<std::endl; }

    /* Won't work till C++17*/
    /* inline Eigen::IOFormat numpy_format(Eigen::StreamPrecision, 0, ", ", ";\n", "[", "]", "[", "]"); */

    inline Eigen::IOFormat setNumpyFormat() {
        return Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
    }
}

#endif
