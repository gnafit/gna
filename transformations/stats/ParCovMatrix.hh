#pragma once
#include "GNAObject.hh"
#include <vector>
#include "UncertainParameter.hh"


class ParCovMatrix: public GNAObject,
                 public TransformationBind<ParCovMatrix> {
    public:
        ParCovMatrix() {
            transformation_("unc_matrix")
                .output("unc_matrix")
                .types([](ParCovMatrix* obj, TypesFunctionArgs fargs){
                       fargs.rets[0] = DataType().points().shape(obj->m_pars.size(), obj->m_pars.size());})
                .func(&ParCovMatrix::FillMatrix);
        };

        ParCovMatrix(std::vector<GaussianParameter<double>*> pars): ParCovMatrix() {m_pars = pars;};

        void append(GaussianParameter<double>* par) {m_pars.push_back(par);};
        void materialize();

    private:
        std::vector<GaussianParameter<double>*> m_pars;
        void FillMatrix(FunctionArgs fargs);
};

