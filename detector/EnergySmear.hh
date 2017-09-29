#ifndef ENERGYSMEAR_H
#define ENERGYSMEAR_H

#include "GNAObject.hh"

class EnergySmear: public GNASingleObject,
                   public Transformation<EnergySmear> {
public:
  EnergySmear( bool triangular=false );

private:
  void calcSmear(Args args, Rets rets);
  void calcSmearTriangular(Args args, Rets rets);

  DataType m_datatype;
};

#endif // ENERGYSMEAR_H
