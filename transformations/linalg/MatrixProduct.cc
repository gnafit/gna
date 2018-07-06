#include "MatrixProduct.hh"
#include "fmt/format.h"



void MatrixProduct::multiply(SingleOutput& out) {
    t_["product"].input(out);
}

 /* Check that matrices have correct shape for computing matrix product.  */
void MatrixProduct::checkTypes(Atypes args, Rtypes rets) {
    if (args.size() == 1) return;
    for (size_t i=1; i<args.size(); ++i) {
        auto& prev = args[i-1]; 
        auto& cur = args[i];
        if ((prev.shape.size()!=2) ||(cur.shape.size()!=2)) {
            throw std::runtime_error("Trying to use something different from matrices in a matrix product");
        }
        if (prev.shape.back() != cur.shape.front()) {
                auto msg = fmt::format("Shapes of matrices doesn't match: ({0},{1})x({2},{3})",
                                        prev.shape[0] % prev.shape[1] % cur.shape[0] % cur.shape[1]);
                throw std::runtime_error(msg);
        }
    }
    rets[0] = DataType().points().shape(args[0].shape[0], args[args.size()-1].shape[1]);
    
}

void MatrixProduct::product(Args args, Rets rets) {
    Eigen::MatrixXd prod = args[0].mat;
    for (size_t i=1; i < args.size(); ++i) {
        prod *= args[i].mat;
    }

    assert(rets[0].mat.cols() == prod.cols() && rets[0].mat.rows() == prod.rows());

    rets[0].mat = prod;
}