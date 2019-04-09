#include "Convolution.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

template<typename FloatType>
GNA::GNAObjectTemplates::ConvolutionT<FloatType>::ConvolutionT()
{
    this->transformation_("normconvolution")
        .input("fcn")
        .input("weights")
        .output("result")
        .output("product")
        .types(new CheckNdimT<FloatType>(1))
        .types(new CheckSameTypesT<FloatType>({0,-1}, "shape"))
        .types(new SetPointsT<FloatType>(1, {0,0}), new PassTypeT<FloatType>(0, {1,1}))
        .func(&ConvolutionType::convolute);

}

template<typename FloatType>
GNA::GNAObjectTemplates::ConvolutionT<FloatType>::ConvolutionT(const std::string& scale) :
ConvolutionT()
{
  m_scale=variable<FloatType>();
  this->variable_(&m_scale.value(), scale);
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ConvolutionT<FloatType>::convolute(ConvolutionT<FloatType>::FunctionArgs& fargs){
    auto& arg0    = fargs.args[0].x;
    auto& weights = fargs.args[1].x;
    auto& result  = fargs.rets[0].x;
    auto& product = fargs.rets[1].x;

    product = arg0*weights;
    if(m_scale){
      result(0) = m_scale.value().value()*product.sum();
    }else{
      result(0) = product.sum();
    }
}

template class GNA::GNAObjectTemplates::ConvolutionT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::ConvolutionT<float>;
#endif
