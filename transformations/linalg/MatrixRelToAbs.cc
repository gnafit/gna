#include "MatrixRelToAbs.hh"
#include "fmt/format.h"
#include <stdexcept>


MatrixRelToAbs::MatrixRelToAbs() {
    transformation_("product")
        .output("product")
        .input("spectra")
        .input("matrix")
        .types(&MatrixRelToAbs::checkTypes)
        .func(&MatrixRelToAbs::product);
        }

void MatrixRelToAbs::multiply(SingleOutput& out) const {
    t_["product"].input(out);
}

void MatrixRelToAbs::checkTypes(TypesFunctionArgs& fargs) const {
 /* Check that input vector and matrix have correct shapes to be multiplied   */
    auto& args = fargs.args;
    auto& rets = fargs.rets;

    if (args.size() == 1) {
        return;
    };

    if (args.size() > 2) {
        throw std::runtime_error(fmt::format("Expecting only two inputs, passed {}", args.size()));
    }

    auto first = args[0];
    auto second = args[1];

    if (second.shape[0] != second.shape[1]) {
        throw std::runtime_error(fmt::format("Matrix is not squared: {0}x{1}", second.shape[0], second.shape[1]));
    }

    if (first.shape.back() != second.shape.front()) {
            throw std::runtime_error(fmt::format("Shape mismatch: ({0})x({1},{2})",
                                    first.shape[0], second.shape[0], second.shape[1]));
    }
    rets[0] = DataType().points().shape(second.shape[0], second.shape[1]);
}

void MatrixRelToAbs::product(FunctionArgs& fargs) {
    auto& args = fargs.args;

    auto& array = args[0].arr;
    auto& matrix = args[1].arr2d;
    auto& matrix_out = fargs.rets[0].arr2d;

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix_out(i, j) = array(i) * array(j) * matrix(i, j);
        }
    }
}
