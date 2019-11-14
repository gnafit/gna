#include "ProductBC.hh"
#include "GNAObject.hh"
#include "TypesFunctions.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    ProductBCT<FloatType>::ProductBCT() {
      this->transformation_("product")
        .output("product")
        .types(TypesFunctions::ifSameShapeOrSingle, TypesFunctions::passNonSingle<0,0>)
        .func([](typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs) {
            auto& args=fargs.args;
            auto& ret=fargs.rets[0].x;
            FloatType factor=1.0;
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
    	;
    }

    /**
     * @brief Construct ProductBC from vector of SingleOutput instances
     */
    template<typename FloatType>
    ProductBCT<FloatType>::ProductBCT(const typename OutputDescriptor::OutputDescriptors& outputs) : ProductBCT<FloatType>(){
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
    InputDescriptorT<FloatType,FloatType> ProductBCT<FloatType>::multiply(SingleOutput &out) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(out));
    }

    /**
     * @brief Add an input by name and leave unconnected.
     * @param name -- a name for the new input.
     * @return InputDescriptor instance for the newly created input.
     */
    template<typename FloatType>
    InputDescriptorT<FloatType,FloatType> ProductBCT<FloatType>::add_input(const char* name) {
      return InputDescriptorT<FloatType,FloatType>(this->t_[0].input(name));
    }
  }
}
template class GNA::GNAObjectTemplates::ProductBCT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  //template class GNA::GNAObjectTemplates::ProductBCT<float>;
#endif
