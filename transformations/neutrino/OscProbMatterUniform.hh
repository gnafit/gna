#ifndef OSCPROBUNIFORMMATTER_H
#define OSCPROBUNIFORMMATTER_H

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "OscProbPMNS.hh"

class  OscProbMatter: public OscProbPMNSBase,
                      public TransformationBlock<OscProbMatter> {
public:
    OscProbMatter(Neutrino from, Neutrino to);

    void calcOscProb(Args args, Rets rets);

protected:
    variable<double> m_L;
    variable<double> m_rho;
    Neutrino m_from;
    Neutrino m_to;
};

#endif //OSCPROBUNIFORMMATTER_H
