#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

/**
 * @brief Transformation to calculate the value of Exp(x)
 *
 * Inputs:
 *   - exp.points
 *   - exp.result
 *
 * @author Maxim Gonchar
 * @date 27.02.2018
 */
class Exp: public GNASingleObject,
           public TransformationBind<Exp> {
public:
    Exp();                               ///< Constructor.

    void calculate(FunctionArgs& fargs); ///< Calculate the value of function.
    void calc_gpu(FunctionArgs& fargs); ///< Calculate the value of function on GPU.
protected:
};
