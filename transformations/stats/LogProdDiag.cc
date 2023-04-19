#include "LogProdDiag.hh"

using namespace Eigen;

LogProdDiag::LogProdDiag(double scale) : m_scale{scale} {
    transformation_("logproddiag")
        .output("logproddiag")
        .types(&LogProdDiag::checkTypes)
        .func(&LogProdDiag::calculateLogProdDiag)
        ;
}

void LogProdDiag::add(SingleOutput &lcov) {
    t_["logproddiag"].input(lcov);
}

void LogProdDiag::checkTypes(TypesFunctionArgs fargs) {
    auto& args=fargs.args;
    auto& rets=fargs.rets;
    for (size_t i = 0; i < args.size(); i++) {
        auto& lcov = args[i];
        switch(lcov.shape.size()){
            case 1:
                /// lcov: uncorrelated uncertainties (first power, not quadratic)
                break;
            case 2:
                /// lcov: L - lower triangular decomposition of covariance matrix
                if (lcov.shape[0] != lcov.shape[1]) {
                    throw rets.error(rets[0], "incompatible covmat shape");
                }
                break;
            default:
                throw rets.error(rets[0], "invalid dimension (errors input)");
                break;
        }
    }

    rets[0] = DataType().points().shape(1);
}

void LogProdDiag::calculateLogProdDiag(FunctionArgs fargs) {
    auto& args=fargs.args;
    double res=0.0;
    for (size_t i = 0; i < args.size(); i++) {
        auto& errors = args[i];
        switch(errors.type.shape.size()){
            case 1:
                res+=errors.arr.log().sum();
                break;
            case 2:
                res+=errors.mat.diagonal().array().log().sum();
                break;
            default:
                break;
        }
    }
    fargs.rets[0].arr(0)=m_scale*res;
}
