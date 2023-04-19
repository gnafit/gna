#include "CovarianceToyMC.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>

using GNA::MatrixFormat;

CovarianceToyMC::CovarianceToyMC(bool autofreeze, MatrixFormat matrix_format) :
GNAObjectBindkN("toymc", {"theory", "cov_L"}, "toymc", 0, 0, 0),
m_autofreeze( autofreeze ),
m_permit_diagonal{matrix_format==MatrixFormat::PermitDiagonal}
{
    this->add_transformation();
    this->add_inputs();
    this->set_open_input();

    GNA::Random::register_callback( [this]{ this->m_distr.reset(); } );
}

void CovarianceToyMC::nextSample() {
    for (size_t i = 0; i < this->transformations.size(); ++i) {
        auto trans = this->transformations[i];
        trans.unfreeze();
        trans.taint();
    }
}

void CovarianceToyMC::calcTypes(TypesFunctionArgs fargs) {
    auto& args=fargs.args;
    auto& rets=fargs.rets;
    if (args.size()%2 != 0) {
        throw args.undefined();
    }
    for (size_t i = 0; i < args.size(); i+=2) {
        auto& central = args[i+0];
        auto& cov_L = args[i+1];
        if (central.shape.size() != 1) {
            throw rets.error(rets[0], "expect 1d array of central values");
        }

        switch(cov_L.shape.size()){
            case 2:
                if (cov_L.shape[0] != cov_L.shape[1]) {
                    throw args.error(cov_L, "CovarianceToyMC expects a square covariance matrix (decomposition), got rectangular");
                }
                break;
            case 1:
                if(!m_permit_diagonal){
                    throw fargs.args.error(cov_L, "CovarianceToyMC input should be a matrix. Enable `permit_diagonal` to work with 1d inputs");
                }
                break;
            default:
                throw args.error(cov_L);
        }
        if (cov_L.shape[0] != central.shape[0]) {
            throw rets.error(rets[0], "central and covariance/diagonal shapes are not consistent");
        }
    }
}

void CovarianceToyMC::calcToyMC(FunctionArgs fargs) {
    auto& args=fargs.args;
    auto& rets=fargs.rets;

    for (size_t i = 0; i < args.size(); i+=2) {
        auto& arg_out = rets[i/2];
        auto& out = rets[i/2].arr;
        auto& arg_central = args[i+0];
        for (int j = 0; j < out.size(); ++j) {
            out(j) = m_distr( GNA::Random::gen() );
        }
        auto& arg_L = args[i+1];
        switch(arg_L.type.shape.size()){
            case 2:
                out = (arg_central.vec + arg_L.mat.triangularView<Eigen::Lower>()*arg_out.vec).eval();
                break;
            case 1:
                out = (arg_central.arr + arg_L.arr*out).eval();
                break;
            default: // Should not happen
                throw rets.error("CovarianceToyMC: invalid input dimension");
                break;
        }
    }
    if(m_autofreeze) {
        rets.untaint();
        rets.freeze();
    }
}

TransformationDescriptor CovarianceToyMC::add_transformation(const std::string& name){
    this->transformation_(new_transformation_name(name))
        .types(new TypeClasses::PassEachTypeT<double>({0,-1,2}))
        .types(&CovarianceToyMC::calcTypes)
        .func(&CovarianceToyMC::calcToyMC);

    reset_open_input();
    return transformations.back();
}
