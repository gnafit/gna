#include "Sum.hh"
#include "TypesFunctions.hh"

/**
 * @brief Constructor.
 */
Sum::Sum() {
  transformation_("sum")                               ///< Define the transformation `sum`:
    .output("sum")                                     ///<   - the transformation `sum` has a single output `sum`
    .types(                                            ///<   - provide type checking functions:
           TypesFunctions::ifSame,                     ///<     * check that inputs have the same type and size
           TypesFunctions::pass<0>                     ///<     * the output type is derived from the first input type
           )                                           ///<
    .func([](FunctionArgs& fargs) {                    ///<   - provide the calculation function:
        auto& args=fargs.args;                         ///<     * extract transformation inputs
        auto& ret=fargs.rets[0].x;                     ///<     * extract transformation output
        ret = args[0].x;                               ///<     * assign (copy) the first input to output
        for (size_t j = 1; j < args.size(); ++j) {     ///<     * iteratively add all the other inputs
          ret += args[j].x;                            ///<
        }                                              ///<
      });                                              ///<
}

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
