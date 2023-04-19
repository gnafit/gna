#pragma once

#include "GNAObject.hh"
#include <TF2.h>

/**
 * @brief A wrapper to use TF2 for vectorized computation
 *
 * Inputs:
 *   - tf2.arg1
 *   - tf2.arg2
 *   - tf2.result
 *
 * @author Maxim Gonchar
 * @date 15.02.2021
 */
class TransformationTF2: public GNASingleObject,
                         public TransformationBind<TransformationTF2> {
public:
    TransformationTF2(TF2* fcn);                         ///< Constructor.
    TransformationTF2(TF2* fcn, OutputDescriptor& x, OutputDescriptor& y);

    void calculate(FunctionArgs& fargs); ///< Calculate the value of function.
protected:
    TF2* m_fcn;
};
