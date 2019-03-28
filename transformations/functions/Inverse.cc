#include "Inverse.hh"
#include "TypesFunctions.hh"

/**
 * @brief Constructor.
 */
Inverse::Inverse() {
  transformation_("inverse")                             ///< Define the transformation `ratio`:
    .input("denom")                                   ///<   - input: denominator
    .output("Inverse")                                   ///<   - the transformation `ratio` has a single output `ratio`
    .types(                                            ///<   - provide type checking functions:
           TypesFunctions::ifSame,                     ///<     * check that inputs have the same type and size
           TypesFunctions::pass<0>                     ///<     * the output type is derived from the first input type
           )                                           ///<
    .func([](FunctionArgs& fargs) {                    ///<   - provide the calculation function:
        auto& args=fargs.args;                         ///<     * extract transformation inputs
        fargs.rets[0].x = 1./args[0].x;         ///<     * compute the ratio
      });                                              ///<
}

/**
 * @brief Construct ratio of top and bottom
 * @param top — nominator
 * @param bottom — denominator
 */
Inverse::Inverse(SingleOutput& bottom) : Inverse() {
  inverse(bottom);
}


/**
 * @brief Bind nomenator, denomenator and return the ratio (output)
 * @param top — nominator
 * @param bottom — denominator
 * @return the ratio output
 */
OutputDescriptor Inverse::inverse(SingleOutput& bottom){
  const auto& t = t_[0];
  const auto& inputs = t.inputs();
  inputs[0].connect(bottom.single());
  return OutputDescriptor(t.outputs()[0]);
}
