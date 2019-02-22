#pragma once

#include "GNAObject.hh"
#include <vector>


class PolynomialNonlinearity: public GNASingleObject,
                              public Transformation<PolynomialNonlinearity> {
    public:
        explicit PolynomialNonlinearity(int poly_order);

    protected:
        void computeNewBins(Args args, Rets rets) noexcept;
        int m_poly_order;
        std::vector<variable<double>> m_coeffs;
};
