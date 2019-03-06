#pragma once

#include "GNAObject.hh"
#include <vector>

class MixedNonlinearity: public GNASingleObject,
                         public TransformationBind<MixedNonlinearity> {
    public:
        explicit MixedNonlinearity();

    protected:
        void computeNewBins(FunctionArgs& fargs) noexcept;
        variable<double> m_alpha, m_beta;
};
