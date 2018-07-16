#include "Sum.hh"
#include "TypesFunctions.hh"
#include "GNAObject.hh" 


/**
 * @brief Constructor.
 */
Sum::Sum(bool gpu = false) : isgpu(gpu) {
  transformation_("sum")                               ///< Define the transformation `sum`:
    .output("sum")                                     ///<   - the transformation `sum` has a single output `sum`
    .types(                                            ///<   - provide type checking functions:
           TypesFunctions::ifSame,                     ///<     * check that inputs have the same type and size
           TypesFunctions::pass<0>                     ///<     * the output type is derived from the first input type
           )                                           ///<
    .func(&Sum::makesum)
#ifdef GNA_CUDA_SUPPORT                                                  ///<
    .setEntryLocation(gpu? DataLocation::Device : DataLocation::Host)
#endif
    ;
  }

void Sum::makesum(Args args, Rets rets) {
#ifdef GNA_CUDA_SUPPORT
  if (isgpu) gpu_sum(args, rets);
  else
#endif
  cpu_sum(args, rets);
}

void Sum::cpu_sum(Args args, Rets rets) {
  rets[0].x = args[0].x;                         ///<     * assign (copy) the first input to output
  for (size_t i = 1; i < args.size(); i++) {     ///<     * iteratively add all the other inputs
    rets[0].x += args[i].x;    
  }
}

#ifdef GNA_CUDA_SUPPORT
  void Sum::gpu_sum(Args args, Rets rets) {
    rets[0].gpuArr->setByDeviceArray(args[0].gpuArr->devicePtr);
    size_t n = args.size();
//    for (size_t i = 0; i < rets[0].gpuArr->arrSize; i++) {
//      printf("%f\n", rets[0].x[i]);
//    }

    for (size_t i = 1; i < n; i++) {
      *(rets[0].gpuArr) += *(args[i].gpuArr);
    }

    rets[0].gpuArr->dump();
    //for (size_t i = 0; i < rets[0].gpuArr->arrSize; i++) {
      //printf("%f\n", rets[0].x[i]);
    //}
  }
#endif

/**
 * @brief Add an input and connect it to the output.
 *
 * The input name is derived from the output name.
 *
 * @param out -- a SingleOutput instance.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor Sum::add(SingleOutput &out) {
  return InputDescriptor(t_["sum"].input(out));
}

/**
 * @brief Add an input by name and leave unconnected.
 * @param name -- a name for the new input.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor Sum::add(const char* name) {
  return InputDescriptor(t_["sum"].input(name));
}
