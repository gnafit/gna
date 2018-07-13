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
        void checkTypes(Atypes args, Rtypes rets);
        void product(FunctionArgs fargs);
};

