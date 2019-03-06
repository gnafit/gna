#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

/**
 * @brief Transformation to calculate the value of Cos(x)
 *
 * Inputs:
 *   - cos.points
 *   - cos.result
 *
 * @author Maxim Gonchar
 * @date 27.02.2018
 */
class Cos: public GNASingleObject,
           public TransformationBind<Cos> {
public:
    Cos();                              ///< Constructor.
    Cos(OutputDescriptor& output);

    void calculate(FunctionArgs& fargs); ///< Calculate the value of function.
protected:
};
