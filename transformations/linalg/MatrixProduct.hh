#pragma once

#include "GNAObject.hh"

class MatrixProduct: public GNAObject, public TransformationBind<MatrixProduct> {
    public:
        MatrixProduct() {
            transformation_("product")
                .output("product")
                .types(&MatrixProduct::checkTypes)
                .func(&MatrixProduct::product);
        }
    void multiply(SingleOutput& single);
    private:
        void checkTypes(TypesFunctionArgs& fargs);
        void product(FunctionArgs& fargs);
};

