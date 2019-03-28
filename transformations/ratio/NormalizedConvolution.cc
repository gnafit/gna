#include "NormalizedConvolution.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

template<typename FloatType>
GNA::GNAObjectTemplates::NormalizedConvolutionT<FloatType>::NormalizedConvolutionT()
{
    this->transformation_("normconvolution")
        .input("fcn")
        .input("weights")
        .output("result")
        .output("product")
        .types(new CheckNdimT<FloatType>(1))
        .types(new CheckSameTypesT<FloatType>({0,-1}, "shape"))
        .types(new SetPointsT<FloatType>(1, {0,0}), new PassTypeT<FloatType>(0, {1,1}))
        .func(&NormalizedConvolutionType::convolute);

}

template<typename FloatType>
void GNA::GNAObjectTemplates::NormalizedConvolutionT<FloatType>::convolute(NormalizedConvolutionT<FloatType>::FunctionArgs& fargs){
    auto& arg0    = fargs.args[0].x;
    auto& weights = fargs.args[1].x;
    auto& result  = fargs.rets[0].x;
    auto& product = fargs.rets[1].x;

    product = arg0*weights;
    result(0) = product.sum()/weights.sum();
}

template class GNA::GNAObjectTemplates::NormalizedConvolutionT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::NormalizedConvolutionT<float>;
#endif
