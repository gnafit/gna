#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

/**
 * @brief Normalize the histogram or a subhistogram so its integral equals to 1.
 *
 * Transformations, inputs and outputs:
 *   - [in]  normalize.inp
 *   - [out] normalize.out
 *
 * @author Maxim Gonchar
 * @date 02.2018
 */
class Normalize: public GNASingleObject,
                 public TransformationBind<Normalize> {
public:
    Normalize();                                    ///< Default constructor.
    Normalize(size_t start, size_t length);         ///< Subhistogram normalization constructor.

    void doNormalize(FunctionArgs fargs);           ///< Normalize the whole histogram.
    void doNormalize_segment(FunctionArgs fargs);   ///< Normalize subhistogram.

protected:
    void checkLimits(TypesFunctionArgs fargs);      ///< typesFunction to check histogram limits.

    size_t m_start;                                 ///< Sub-histogram first bin.
    size_t m_length;                                ///< Sub-histogram length.
};
