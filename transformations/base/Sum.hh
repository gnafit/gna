#pragma once

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
  bool isgpu = false;
  Sum(bool gpu);            
  void makesum(Args args, Rets rets);
  void cpu_sum(Args args, Rets rets);
#ifdef GNA_CUDA_SUPPORT
  void gpu_sum(Args args, Rets rets);
#endif
  InputDescriptor add(SingleOutput &data);   ///< Add an input and connect it to the output.
  InputDescriptor add(const char* name);     ///< Add an input by name and leave unconnected.
};
