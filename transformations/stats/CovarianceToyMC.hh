#pragma once

#include "Random.hh"
#include "GNAObject.hh"
#include "GNAObjectBindkN.hh"
#include "DataEnums.hh"

class CovarianceToyMC: public GNAObjectBindkN,
                       public TransformationBind<CovarianceToyMC> {
public:
    CovarianceToyMC(bool autofreeze=true, GNA::MatrixFormat matrix_format=GNA::MatrixFormat::Regular);

    void add(SingleOutput& theory, SingleOutput &cov) { add_inputs(SingleOutputsContainer({&theory, &cov}));  }
    void nextSample();

    void reset() { m_distr.reset(); }

    TransformationDescriptor add_transformation(const std::string& name="");

protected:
    void calcTypes(TypesFunctionArgs fargs);
    void calcToyMC(FunctionArgs fargs);

    std::normal_distribution<> m_distr;

    bool m_autofreeze;
    bool m_permit_diagonal=false;
};
