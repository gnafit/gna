#include "MatVecProduct.hh"
#include "fmt/format.h"
#include <stdexcept>


MatVecProduct::MatVecProduct() {
    transformation_("product")
        /* .input("first")
         * .input("second") */
        .output("product")
        .types(&MatVecProduct::checkTypes)
        .func(&MatVecProduct::product);
        }

MatVecProduct::MatVecProduct(SingleOutput& first, SingleOutput& second): MatVecProduct() {
    this->multiply(first);
    this->multiply(second);
}

void MatVecProduct::multiply(SingleOutput& out) {
    t_["product"].input(out);
}

void MatVecProduct::checkTypes(TypesFunctionArgs& fargs) {
 /* Check that input vector and matrix have correct shapes to be multiplied   */
    auto& args=fargs.args;
    auto& rets=fargs.rets;

    if (args.size()==1) {
        return;
    };

    if (args.size()>2) {
        throw std::runtime_error(fmt::format("Expecting only two inputs, passed {}", args.size()));
    };

    auto first = args[0];
    auto second = args[1];

    std::string exception_msg{};
    if (first.shape.size() == 1) {
        m_vec_pos = 0;
        exception_msg = fmt::format("Shape mismatch for vector-matrix product: ({0})x({1},{2})",
                                        first.shape[0],  second.shape[0], second.shape[1]);
    } else {
        m_vec_pos = 1;
        exception_msg = fmt::format("Shape mismatch for matrix-vector product: ({0}, {1})x({2})",
                                first.shape[0],  first.shape[1], second.shape[0]);
    }

    if (first.shape.back() != second.shape.front()) {
            throw std::runtime_error(exception_msg);
    }
    if (m_vec_pos == 0) {
        rets[0] = DataType().points().shape(args[1].shape[1]);
    } else {
        rets[0] = DataType().points().shape(args[0].shape[0]);
    }

}

void MatVecProduct::product(FunctionArgs& fargs) {
    auto& args=fargs.args;
    auto& ret=fargs.rets[0].vec;

    if (m_vec_pos == 0) {
        ret.noalias() = args[0].vec.transpose()*args[1].mat;
    } else {
        ret.noalias() = args[0].mat*args[1].vec;
    }
}
