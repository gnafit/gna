#pragma once

#include "GNAObject.hh"

class MatrixRelToAbs: public GNAObject, public TransformationBind<MatrixRelToAbs> {
    public:
        MatrixRelToAbs();
        void multiply(SingleOutput& single) const;
    private:
        void checkTypes(TypesFunctionArgs& fargs) const;
        void product(FunctionArgs& fargs);
};
