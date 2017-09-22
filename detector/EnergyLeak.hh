#ifndef ENERGYLEAK_H
#define ENERGYLEAK_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyLeak: public GNAObject,
                  public Transformation<EnergyLeak> {
public:
  EnergyLeak( size_t n, double* mat_column_major );

  Eigen::MatrixXd& getMatrix() { return m_matrix; }
private:
  void fillCache();
  void calcSmear(Args args, Rets rets);

  DataType m_datatype;

  size_t m_size;
  Eigen::MatrixXd m_matrix;
};

#endif // ENERGYLEAK_H
