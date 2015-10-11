#ifndef ENERGYRESOLUTION_H
#define ENERGYRESOLUTION_H

#include <vector>

#include "GNAObject.hh"

class EnergyResolution: public GNAObject,
                        public Transformation<EnergyResolution> {
public:
  TransformationDef(EnergyResolution)

  EnergyResolution();

private:
  double relativeSigma(double Etrue);
  double resolution(double Etrue, double Erec);
  void fillCache();
  void calcSmear(Args args, Rets rets);

  variable<double> m_a, m_b, m_c;

  DataType m_datatype;

  size_t m_size;
  std::vector<double> m_rescache;
  std::vector<int> m_cacheidx;
  std::vector<int> m_startidx;
};

#endif // ENERGYRESOLUTION_H
