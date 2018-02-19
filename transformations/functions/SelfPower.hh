#ifndef SELFPOWER_H
#define SELFPOWER_H

#include "GNAObject.hh"
#include "Statistic.hh"

/**
 * @brief Transformation to calculate the value of (x/a)^(±x/a) function.
 *
 * Transformations, inputs and outputs:
 *   - selfpower
 *     * [in]  selfpower.points
 *     * [out] selfpower.result
 *   - selfpower_inv
 *     * [in]  selfpower_inv.points
 *     * [out] selfpower_inv.result
 *
 * @author Maxim Gonchar
 * @date
 */
class SelfPower: public GNAObject,
                 public TransformationBlock<SelfPower> {
public:
    SelfPower(const char* scalename="sp_scale"); ///< Constructor.

    void calculate(Args args, Rets rets);        ///< Calculate the value of function with positive power.
    void calculate_inv(Args args, Rets rets);    ///< Calculate the value of function with negative power.
protected:
    variable<double> m_scale;                    ///< The scale (a) variable.
};

#endif
