#pragma once

#include "GNAObject.hh"

class LogProdDiag: public GNASingleObject,
                   public TransformationBind<LogProdDiag> {
public:
    LogProdDiag(double scale=2.0);
    void add(SingleOutput &cov_l);

    void checkTypes(TypesFunctionArgs fargs);
    void calculateLogProdDiag(FunctionArgs fargs);

protected:
    double m_scale=2.0;
};
