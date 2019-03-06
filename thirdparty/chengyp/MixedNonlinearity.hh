#pragma once

#include "GNAObject.hh"
#include <vector>


class MixedNonlinearity: public GNASingleObject,
                              public Transformation<MixedNonlinearity> {
    public:
        explicit MixedNonlinearity();

    protected:
        void computeNewBins(Args args, Rets rets) noexcept;
        variable<double> m_alpha, m_beta;
};
