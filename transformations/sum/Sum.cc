#include "Sum.hh"
#include "TypesFunctions.hh"
#include "GNAObject.hh" 

#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"                             
#include "DataLocation.hh"
#endif

namespace GNA {
  namespace GNAObjectTemplates {
#ifdef GNA_CUDA_SUPPORT
    template<typename FloatType>
    void SumT<FloatType>::sum_ongpu(FunctionArgs& fargs) {
        fargs.args.touch();
        auto& gpuargs=fargs.gpu;
        gpuargs->provideSignatureDevice();
        auto** source=gpuargs->args;
        auto** dest  =gpuargs->rets;
        cusum(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
    }
#endif
    /**
     * @brief Constructor.
     */
    template<typename FloatType>
    SumT<FloatType>::SumT() {
      this->transformation_("sum")                               ///< Define the transformation `sum`:
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
          })
         .func("gpu", &SumT<FloatType>::sum_ongpu, DataLocation::Device)
    	;                                              ///<
    }
    
    /**
     * @brief Construct Sum from vector of SingleOutput instances
     */
    template<typename FloatType>
    SumT<FloatType>::SumT(const OutputDescriptor::OutputDescriptors& outputs) : SumT(){
      for(auto& output : outputs){
        this->add(*output);
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
    InputDescriptorT<FloatType,FloatType> SumT<FloatType>::add(SingleOutput &out) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(out));
    }
    
    /**
     * @brief Add an input by name and leave unconnected.
     * @param name -- a name for the new input.
     * @return InputDescriptor instance for the newly created input.
     */
    template<typename FloatType>
    InputDescriptorT<FloatType,FloatType> SumT<FloatType>::add_input(const char* name) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(name));
    }
    
/*    void sum_ongpu(FunctionArgs& fargs) {
    	fargs.args.touch();
    	auto& gpuargs=fargs.gpu;
    //	gpuargs->provideSignatureDevice();
    	auto** source=gpuargs->args;
            auto** dest  =gpuargs->rets;
    	cusum(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
    }
*/
  }
}
template class GNA::GNAObjectTemplates::SumT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::SumT<float>;
#endif
