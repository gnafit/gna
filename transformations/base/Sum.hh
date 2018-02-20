#ifndef SUM_H
#define SUM_H

#include "GNAObject.hh"

/**
 * @brief Calculate the element-wise sum of the inputs.
 *
 * Outputs:
 *   - `sum.sum` -- the result of a sum.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class Sum: public GNASingleObject,
           public TransformationBind<Sum> {
public:
  Sum();                                     ///< Constructor.
  InputDescriptor add(SingleOutput &data);   ///< Add an input and connect it to the output.
  InputDescriptor add(const char* name);     ///< Add an input by name and leave unconnected.
};

#endif // SUM_H
