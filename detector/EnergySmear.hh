#ifndef ENERGYSMEAR_H
#define ENERGYSMEAR_H

#include "GNAObject.hh"

class EnergySmear: public GNASingleObject,
                   public Transformation<EnergySmear> {
public:
  EnergySmear( bool upper=false );

private:
  void calcSmear(Args args, Rets rets);
  void calcSmearUpper(Args args, Rets rets);
};

#endif // ENERGYSMEAR_H
