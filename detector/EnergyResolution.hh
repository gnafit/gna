#ifndef ENERGYRESOLUTION_H
#define ENERGYRESOLUTION_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyResolution: public GNAObject,
                        public Transformation<EnergyResolution> {
public:
  EnergyResolution();

private:
  double relativeSigma(double Etrue) const noexcept;
  double resolution(double Etrue, double Erec) const noexcept;
  void fillCache();
  void calcSmear(Args args, Rets rets);

  variable<double> m_a, m_b, m_c;

  DataType m_datatype;
  

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;
};

#endif // ENERGYRESOLUTION_H
