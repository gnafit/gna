#pragma once
#include "GNAObject.hh"
#include <vector>
#include "UncertainParameter.hh"


class ParCovMatrix: public GNAObject,
                 public Transformation<ParCovMatrix> {
    public:
        ParCovMatrix() {
            transformation_(this, "unc_matrix")
                .output("unc_matrix")
                .types([](ParCovMatrix* obj, Atypes /*args*/, Rtypes rets){
    rets[0] = DataType().points().shape(obj->m_pars.size(), obj->m_pars.size());})
                .func(&ParCovMatrix::FillMatrix);
        };

        ParCovMatrix(std::vector<Parameter<double>*> pars): ParCovMatrix() {m_pars = pars;};

        void append(Parameter<double>* par) {m_pars.push_back(par);};
        void materialize();

    private:
        std::vector<Parameter<double>*> m_pars;
        void FillMatrix(Args args, Rets rets);

};

