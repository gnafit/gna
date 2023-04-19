#pragma once

#include "GNAObject.hh"

class MatVecProduct: public GNAObject, public TransformationBind<MatVecProduct> {
    public:
        MatVecProduct();
        MatVecProduct(SingleOutput& first, SingleOutput& second);
        void multiply(SingleOutput& single);
    private:
        void checkTypes(TypesFunctionArgs& fargs);
        void product(FunctionArgs& fargs);
        size_t m_vec_pos;
};
