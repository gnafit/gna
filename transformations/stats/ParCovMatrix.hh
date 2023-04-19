#pragma once
#include "GNAObject.hh"
#include <vector>
#include "UncertainParameter.hh"
#include "DataEnums.hh"

class ParCovMatrix: public GNAObject,
                    public TransformationBind<ParCovMatrix> {
public:
    ParCovMatrix(GNA::MatrixFormat matrix_format=GNA::MatrixFormat::Regular);
    ParCovMatrix(std::vector<GaussianParameter<double>*> pars, GNA::MatrixFormat matrix_format=GNA::MatrixFormat::Regular):
        ParCovMatrix(matrix_format) {m_pars = pars; materialize(); }

    void append(GaussianParameter<double>* par);
    void materialize();

private:
    void Types(TypesFunctionArgs fargs);
    void FillMatrix(FunctionArgs fargs);

    std::vector<GaussianParameter<double>*> m_pars;
    bool m_permit_diagonal=false;
    bool m_covariance_diagonal=true;
};

