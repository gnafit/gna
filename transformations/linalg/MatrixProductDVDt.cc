#include "MatrixProductDVDt.hh"
#include "TypesFunctions.hh"
#include "fmt/format.h"

MatrixProductDVDt::MatrixProductDVDt() {
    transformation_("product")
        .input("left")
        .input("square")
        .output("product")
        .types(&TypesFunctions::if2d<0>)
        .types(&TypesFunctions::ifSquare<1>,&TypesFunctions::if2d<1>)
        .types(&MatrixProductDVDt::checkTypes)
        .func(&MatrixProductDVDt::product);
}

void MatrixProductDVDt::multiply(SingleOutput& left, SingleOutput& square) {
    auto inputs = transformations["product"].inputs;
    left.single()   >> inputs["left"];
    square.single() >> inputs["square"];
}

 /* Check that matrices have correct shape for computing matrix product.  */
void MatrixProductDVDt::checkTypes(TypesFunctionArgs& fargs) {
    auto& args=fargs.args;
    auto& rets=fargs.rets;
    if (args.size() != 2) return;

    auto& left   = args[0];
    auto& square = args[1];

    if (left.shape.back() != square.shape.front()) {
            auto msg = fmt::format("Shapes of matrices doesn't match: ({0},{1})x({2},{3})",
                                    left.shape[0] % left.shape[1] % square.shape[0] % square.shape[1]);
            throw std::runtime_error(msg);
    }

    rets[0] = DataType().points().shape(left.shape[0], left.shape[1]);

}

void MatrixProductDVDt::product(FunctionArgs& fargs) {
    auto& args=fargs.args;
    auto& ret=fargs.rets[0].mat;

    auto& left   = args[0].mat;
    auto& square = args[1].mat;
    ret = left * square * left.transpose();
}
