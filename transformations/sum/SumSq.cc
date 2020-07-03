#include "SumSq.hh"
#include "GNAObject.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

namespace GNA {
  namespace GNAObjectTemplates {
    /**
     * @brief Constructor.
     */
    template<typename FloatType>
    SumSqT<FloatType>::SumSqT() {
      this->transformation_("sumsq")                            ///< Define the transformation `sumsq`:
        .output("sumsq")                                        ///<   - the transformation `sumsq` has a single output `sumsq`
        .types(                                                 ///<   - provide type checking functions:
               new CheckSameTypesT<FloatType>({0,-1}, "shape"), ///<     * check that inputs have the same type and size
               new PassTypeT<FloatType>(0,{0,0})                ///<     * the output type is derived from the first input type
               )                                                ///<
        .func([](typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs) { ///<   - provide the calculation function:
            auto& args=fargs.args;                              ///<     * extract transformation inputs
            auto& ret=fargs.rets[0].x;                          ///<     * extract transformation output
            ret = args[0].x.square();                           ///<     * assign (copy) the first input squared to output
            for (size_t j = 1; j < args.size(); ++j) {          ///<     * iteratively add all the other inputs
              ret += args[j].x.square();                        ///<     * squared
            }                                                   ///<
          })
    	;                                                       ///<
    }

    /**
     * @brief Construct SumSq from vector of SingleOutput instances
     */
    template<typename FloatType>
    SumSqT<FloatType>::SumSqT(const typename OutputDescriptor::OutputDescriptors& outputs) : SumSqT(){
      for(auto output : outputs){
        this->add(output);
      }
    }

    /**
     * @brief Add an input and connect it to the output.
     *
     * The input name is derived from the output name.
     *
     * @param out -- a SingleOutput instance.
     * @return InputDescriptor instance for the newly created input.
     */
    template<typename FloatType>
    InputDescriptorT<FloatType,FloatType> SumSqT<FloatType>::add(SingleOutput &out) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(out));
    }

    /**
     * @brief Add an input by name and leave unconnected.
     * @param name -- a name for the new input.
     * @return InputDescriptor instance for the newly created input.
     */
    template<typename FloatType>
    InputDescriptorT<FloatType,FloatType> SumSqT<FloatType>::add_input(const char* name) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(name));
    }

/*    void sumsq_ongpu(FunctionArgs& fargs) {
    	fargs.args.touch();
    	auto& gpuargs=fargs.gpu;
    //	gpuargs->provideSignatureDevice();
    	auto** source=gpuargs->args;
            auto** dest  =gpuargs->rets;
    	cusumsq(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
    }
*/
  }
}
template class GNA::GNAObjectTemplates::SumSqT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::SumSqT<float>;
#endif
