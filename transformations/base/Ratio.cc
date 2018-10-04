#include "Ratio.hh"
#include "TypesFunctions.hh"

/**
 * @brief Constructor.
 */
Ratio::Ratio() {
  transformation_("ratio")                             ///< Define the transformation `ratio`:
    .input("top")                                      ///<   - input: nominator
    .input("bottom")                                   ///<   - input: denominator
    .output("ratio")                                   ///<   - the transformation `ratio` has a single output `ratio`
    .types(                                            ///<   - provide type checking functions:
           TypesFunctions::ifSame,                     ///<     * check that inputs have the same type and size
           TypesFunctions::pass<0>                     ///<     * the output type is derived from the first input type
           )                                           ///<
    .func([](FunctionArgs& fargs) {                    ///<   - provide the calculation function:
        auto& args=fargs.args;                         ///<     * extract transformation inputs
        fargs.rets[0].x = args[0].x/args[1].x;         ///<     * compute the ratio
      });                                              ///<
}

