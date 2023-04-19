#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

class Chi2CNPStat: public GNASingleObject,
                   public TransformationBind<Chi2CNPStat>,
                   public Statistic  {
public:
    Chi2CNPStat();

    void add(SingleOutput &theory, SingleOutput &data);
    void calcChi2CNPStat(FunctionArgs fargs);
    void checkTypes(TypesFunctionArgs fargs);

    double value() override {
        return m_transform[0].x[0];
    }

protected:
    Handle m_transform;
};
