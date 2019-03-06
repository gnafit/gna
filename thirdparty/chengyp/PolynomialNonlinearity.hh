#pragma once

#include "GNAObject.hh"
#include <vector>

class PolynomialNonlinearity: public GNASingleObject,
                              public TransformationBind<PolynomialNonlinearity> {
    public:
        explicit PolynomialNonlinearity(int poly_order);

    protected:
        void computeNewBins(FunctionArgs& fargs) noexcept;
        int m_poly_order;
        std::vector<variable<double>> m_coeffs;
};
