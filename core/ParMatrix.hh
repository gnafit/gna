#pragma once
#include "GNAObject.hh"
#include <vector>
#include "UncertainParameter.hh"


class ParMatrix: public GNAObject,
                 public Transformation<ParMatrix> {
    public:
        ParMatrix() {
            transformation_(this, "unc_matrix")
                .output("unc_matrix")
                .types([](ParMatrix* obj, Atypes /*args*/, Rtypes rets){
    rets[0] = DataType().points().shape(obj->m_pars.size(), obj->m_pars.size());})
                .func(&ParMatrix::FillMatrix);
        };
        ParMatrix(std::vector<Parameter<double>*> pars): ParMatrix() {m_pars = pars;};
        void append(Parameter<double>* par) {m_pars.push_back(par);};
        void materialize();

    private:
        std::vector<Parameter<double>*> m_pars;
        void FillMatrix(Args args, Rets rets);

};

