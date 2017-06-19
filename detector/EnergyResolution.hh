#ifndef ENERGYRESOLUTION_H
#define ENERGYRESOLUTION_H

#include <fstream>
#include <vector>

#include "GNAObject.hh"

class EnergyResolution: public GNAObject,
                        public Transformation<EnergyResolution> {
public:
  EnergyResolution();
  ~EnergyResolution();

private:
  double relativeSigma(double Etrue) const noexcept;
  double resolution(double Etrue, double Erec) const noexcept;
  void fillCache();
  void calcSmear(Args args, Rets rets);

  variable<double> m_a, m_b, m_c;

  DataType m_datatype;

  size_t m_size;
  std::vector<double> m_rescache;
  std::vector<int> m_cacheidx;
  std::vector<int> m_startidx;
  std::ofstream m_bench_file;
};

#endif // ENERGYRESOLUTION_H
