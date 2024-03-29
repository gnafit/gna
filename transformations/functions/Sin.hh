#pragma once

#include "GNAObject.hh"

/**
 * @brief Transformation to calculate the value of Sin(x)
 *
 * Inputs:
 *   - sin.points
 *   - sin.result
 *
 * @author Maxim Gonchar
 * @date 27.02.2018
 */
class Sin: public GNASingleObject,
           public TransformationBind<Sin> {
public:
    Sin();                               ///< Constructor.
    Sin(OutputDescriptor& output);

    void calculate(FunctionArgs& fargs); ///< Calculate the value of function.
protected:
};
