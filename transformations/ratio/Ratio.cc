#include "Ratio.hh"
#include "TypeClasses.hh"

using TypeClasses::CheckSameTypesT;
using TypeClasses::PassTypePriorityT;

/**
 * @brief Constructor.
 */
Ratio::Ratio() {
  transformation_("ratio")                               ///< Define the transformation `ratio`:
    .input("top")                                        ///<   - input: nominator
    .input("bottom")                                     ///<   - input: denominator
    .output("ratio")                                     ///<   - the transformation `ratio` has a single output `ratio`
    .types(                                              ///<   - provide type checking functions:
           new CheckSameTypesT<double>({0,-1}, "shape"), ///<     * check that inputs have the same type and size
           new PassTypePriorityT<double>({0,-1},{0,-1})  ///<     * the output type is derived from the earliest input type, histograms with N>1 preferred
           )                                             ///<
    .func([](FunctionArgs& fargs) {                      ///<   - provide the calculation function:
        auto& args=fargs.args;                           ///<     * extract transformation inputs
        fargs.rets[0].x = args[0].x/args[1].x;           ///<     * compute the ratio
      });                                                ///<
}

/**
 * @brief Construct ratio of top and bottom
 * @param top — nominator
 * @param bottom — denominator
 */
Ratio::Ratio(SingleOutput& top, SingleOutput& bottom) : Ratio() {
  divide(top, bottom);
}


/**
 * @brief Bind nomenator, denomenator and return the ratio (output)
 * @param top — nominator
 * @param bottom — denominator
 * @return the ratio output
 */
OutputDescriptor Ratio::divide(SingleOutput& top, SingleOutput& bottom){
  const auto& t = t_[0];
  const auto& inputs = t.inputs();
  inputs[0].connect(top.single());
  inputs[1].connect(bottom.single());
  return OutputDescriptor(t.outputs()[0]);
}
