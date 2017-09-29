#ifndef ENERGYSMEAR_H
#define ENERGYSMEAR_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergySmear: public GNASingleObject,
                   public Transformation<EnergySmear> {
public:
  EnergySmear( size_t n, double* mat_column_major, bool triangular=false );

  Eigen::MatrixXd& getMatrix() { return m_matrix; }
private:
  void fillCache();
  void calcSmear(Args args, Rets rets);
  void calcSmearTriangular(Args args, Rets rets);

  DataType m_datatype;

  size_t m_size;
  Eigen::MatrixXd m_matrix;
};

#endif // ENERGYSMEAR_H
