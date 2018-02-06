#ifndef SELFPOWER_H
#define SELFPOWER_H

#include "GNAObject.hh"
#include "Statistic.hh"

class SelfPower: public GNASingleObject,
          public Transformation<SelfPower> {
public:
    SelfPower(const char* scalename="sp_scale");

    void calculate(Args args, Rets rets);
    void calculate_inv(Args args, Rets rets);
protected:
    variable<double> m_scale;
};

#endif
