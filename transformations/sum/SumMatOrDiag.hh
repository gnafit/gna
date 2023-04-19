#pragma once

#include "GNAObject.hh"

/**
 * @brief Calculate matrix sum of the inputs, vectors are treated as diagonal matrices.
 *
 * Outputs:
 *  - `sum.sum` -- the result of a sum. Square matrix if one of the inputs is a matrix, vector otherwise.
 */

class SumMatOrDiag : public GNASingleObject, public TransformationBind<SumMatOrDiag> {
public:
    SumMatOrDiag();
    SumMatOrDiag(const OutputDescriptor::OutputDescriptors& outputs);

    InputDescriptor add(SingleOutput& data) const;

    void checkTypes(TypesFunctionArgs fargs) const;
    void calculateSum(FunctionArgs& fargs) const;

private:
    void sumVec(FunctionArgs& fargs) const;
    void sumMat(FunctionArgs& fargs) const;
};
