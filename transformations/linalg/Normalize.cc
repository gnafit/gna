#include "Normalize.hh"
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

/**
 * @brief Default constructor.
 *
 * Initializes the transformation for the whole histogram normalization.
 */
Normalize::Normalize() {
    transformation_("normalize")
        .input("inp")
        .output("out")
        .types(Atypes::pass<0>)
        .func(&Normalize::doNormalize)
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
        .types(Atypes::pass<0>, &Normalize::checkLimits)
        .func(&Normalize::doNormalize_segment)
        ;
}

/**
 * @brief Normalize the whole histogram.
 *
 * Divides each bin to the sum of bins.
 */
void Normalize::doNormalize(Args args, Rets rets){
    auto& in=args[0].x;
    rets[0].x=in/in.sum();
}

/**
 * @brief Normalize subhistogram.
 *
 * Divides each bin to the sum of bins in a range [start, start+length-1].
 */
void Normalize::doNormalize_segment(Args args, Rets rets){
    auto& in=args[0].x;
    rets[0].x=in/in.segment(m_start, m_length).sum();
}

/**
 * @brief typesFunction to check histogram limits for subhistogram mode.
 * @exception SourceTypeError in case the input array is not 1d.
 * @exception SourceTypeError in case the start is outside of the data limits.
 * @exception SourceTypeError in case the end is outside of the data limits.
 */
void Normalize::checkLimits(Atypes args, Rtypes rets) {
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
