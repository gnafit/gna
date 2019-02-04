#include "Product.hh"
#include "TypesFunctions.hh"

#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"                             
#include "DataLocation.hh"
#endif

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    ProductT<FloatType>::ProductT() {
      this->transformation_("product")
        .output("product")
        .types(TypesFunctions::ifSame, TypesFunctions::passNonSingle<0,0>) // check ifSame ... was ifSameShapeOrSingle TODO smth?
        .func([](FunctionArgs& fargs) {
            auto& args=fargs.args;
            auto& ret=fargs.rets[0].x;
            double factor=1.0;
            bool secondary=false;
            for (size_t i = 0; i < args.size(); ++i) {
              auto& data=args[i].x;
              if (data.size()!=1) {
                if (secondary) {
                  ret*=data;
                } else {
                  ret=data;
                  secondary=true;
                }
              }
              else{
                factor*=data(0);
              }
            }
            if(!secondary){
              ret=factor;
            }
            else if(factor!=1){
              ret*=factor;
            }
          })
        .func("gpu", &ProductT<FloatType>::product_ongpu, DataLocation::Device)
    	;
    }
    
    /**
     * @brief Construct Product from vector of SingleOutput instances
     */
    template<typename FloatType>
    ProductT<FloatType>::ProductT(const OutputDescriptor::OutputDescriptors& outputs) : ProductT<FloatType>(){
      for(auto& output : outputs){
        this->multiply(*output);
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
    InputDescriptorT<FloatType,FloatType> ProductT<FloatType>::multiply(SingleOutput &out) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(out));
    }
    
    /**
     * @brief Add an input by name and leave unconnected.
     * @param name -- a name for the new input.
     * @return InputDescriptor instance for the newly created input.
     */
    template<typename FloatType>
    InputDescriptorT<FloatType,FloatType> ProductT<FloatType>::add_input(const char* name) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(name));
    }
    
    template<typename FloatType>
    void ProductT<FloatType>::product_ongpu(FunctionArgs& fargs) {
    	fargs.args.touch();
    	auto& gpuargs=fargs.gpu;
    	auto** source=gpuargs->args;
        auto** dest  =gpuargs->rets;
    	cuproduct(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
    }
  }
}
template class GNA::GNAObjectTemplates::ProductT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::ProductT<float>;
#endif
