#pragma once

#include "GNAObject.hh"

/**
 * @brief Compute D V D^T product
 * @author Maxim Gonchar
 * @date 2019.12.02
 */
class MatrixProductDVDt: public GNAObject, public TransformationBind<MatrixProductDVDt> {
public:
    MatrixProductDVDt();
    MatrixProductDVDt(SingleOutput& left, SingleOutput& square) : MatrixProductDVDt() {
        multiply(left, square);
    }
private:
    void multiply(SingleOutput& left, SingleOutput& square);
    void checkTypes(TypesFunctionArgs& fargs);
    void product(FunctionArgs& fargs);
};

