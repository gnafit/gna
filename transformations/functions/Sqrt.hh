#pragma once

#include "GNAObject.hh"

/**
 * @brief Transformation to calculate the value of Sqrt(x)
 *
 * Inputs:
 *   - sqrt.points
 *   - sqrt.result
 *
 * @author Maxim Gonchar
 * @date 22.08.2022
 */
class Sqrt: public GNASingleObject,
            public TransformationBind<Sqrt> {
public:
  Sqrt();                               ///< Constructor.
  Sqrt(OutputDescriptor& output);

  void calculate(FunctionArgs& fargs); ///< Calculate the value of function.
};
