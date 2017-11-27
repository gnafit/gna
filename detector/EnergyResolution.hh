#ifndef ENERGYRESOLUTION_H
#define ENERGYRESOLUTION_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyResolution: public GNASingleObject,
                        public Transformation<EnergyResolution> {
public:
  EnergyResolution( bool single=true );

  double relativeSigma(double Etrue) const noexcept;
  double resolution(double Etrue, double Erec) const noexcept;

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  void add();
  void add(SingleOutput& hist);
private:
  void fillCache();
  void calcSmear(Args args, Rets rets);

  variable<double> m_a, m_b, m_c;

  DataType m_datatype;

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;

  bool m_single; /// restricts EnergyResolution to contain only one transformation
};

#endif // ENERGYRESOLUTION_H
