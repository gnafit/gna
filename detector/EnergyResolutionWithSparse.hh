#ifndef ENERGYRESOLUTIONWITHSPARSE_H
#define ENERGYRESOLUTIONWITHSPARSE_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"
#include <fstream>

class EnergyResolutionWithSparse: public GNAObject,
                        public Transformation<EnergyResolutionWithSparse> {
public:
  EnergyResolutionWithSparse();
  ~EnergyResolutionWithSparse();

private:
  double relativeSigma(double Etrue) const noexcept;
  double resolution(double Etrue, double Erec) const noexcept;
  void fillCache();
  void calcSmear(Args args, Rets rets);

  variable<double> m_a, m_b, m_c;

  DataType m_datatype;
  
  std::ofstream m_bench_file;

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;
};

#endif // ENERGYRESOLUTIONWITHSPARSE_H
