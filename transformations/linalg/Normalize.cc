#include "Normalize.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>
#include <cmath>

#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#include "DataLocation.hh"
#endif 

/**
 * @brief Default constructor.
 *
 * Initializes the transformation for the whole histogram normalization.
 */
Normalize::Normalize() {
    transformation_("normalize")
        .input("inp")
        .output("out")
        .types(TypesFunctions::pass<0>)
        .func(&Normalize::doNormalize)
#ifdef GNA_CUDA_SUPPORT
        .func("gpu", &Normalize::doNormalize_gpu, DataLocation::Device)
#endif
        ;
}

/**
 * @brief Subhistogram normalization constructor.
 *
 * Start and length are defined the same way they are defined for the segment method of Eigen.
 *
 * @param start  -- subhistogram first bin.
 * @param length -- number of bins to normalize to.
 */
Normalize::Normalize(size_t start, size_t length) : m_start{start}, m_length{length} {
    transformation_("normalize")
        .input("inp")
        .output("out")
        .types(TypesFunctions::pass<0>, &Normalize::checkLimits)
        .func(&Normalize::doNormalize_segment)
        ;
}

/**
 * @brief Normalize the whole histogram.
 *
 * Divides each bin to the sum of bins.
 */
void Normalize::doNormalize(FunctionArgs& fargs){
    auto& in=fargs.args[0].x;
    fargs.rets[0].x=in/in.sum();
}

void Normalize::doNormalize_gpu(FunctionArgs& fargs) {
    fargs.args.touch();
    auto& gpuargs = fargs.gpu;
    gpuargs->provideSignatureDevice();
    cunormalize(gpuargs->args, gpuargs->rets, fargs.args[0].arr.size());
}

/**
 * @brief Normalize subhistogram.
 *
 * Divides each bin to the sum of bins in a range [start, start+length-1].
 */
void Normalize::doNormalize_segment(FunctionArgs& fargs){
    auto& in=fargs.args[0].x;
    fargs.rets[0].x=in/in.segment(m_start, m_length).sum();
}

/**
 * @brief typesFunction to check histogram limits for subhistogram mode.
 * @exception SourceTypeError in case the input array is not 1d.
 * @exception SourceTypeError in case the start is outside of the data limits.
 * @exception SourceTypeError in case the end is outside of the data limits.
 */
void Normalize::checkLimits(TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
    auto& dtype = args[0];
    if( dtype.shape.size()!=1u ){
        throw args.error(dtype, "Accept only 1d arrays in case a segment is specified");
    }
    auto length=dtype.shape[0];
    if( m_start>=length ){
        throw args.error(dtype, "Segment start is outside of the data limits");
    }
    if( (m_start+m_length)>length ){
        throw args.error(dtype, "Segment end is outside of the data limits");
    }
}
