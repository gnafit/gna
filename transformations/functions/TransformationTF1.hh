#pragma once

#include "GNAObject.hh"
#include <TF1.h>

/**
 * @brief A wrapper to use TF1 for vectorized computation
 *
 * Inputs:
 *   - tf1.arg
 *   - tf1.result
 *
 * @author Maxim Gonchar
 * @date 15.02.2021
 */
class TransformationTF1: public GNASingleObject,
                         public TransformationBind<TransformationTF1> {
public:
    TransformationTF1(TF1* fcn);                         ///< Constructor.
    TransformationTF1(TF1* fcn, OutputDescriptor& output);

    void calculate(FunctionArgs& fargs); ///< Calculate the value of function.
protected:
    TF1* m_fcn;
};
