#ifndef REBIN_H
#define REBIN_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class Rebin: public GNAObject,
             public Transformation<Rebin> {
public:
  Rebin(size_t n, double* edges, int rounding);

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

private:
  void calcMatrix(Atypes args, Rtypes rets);
  void calcSmear(Args args, Rets rets);

  std::vector<double> m_new_edges;
  int m_round_scale;

  Eigen::SparseMatrix<double> m_sparse_cache;
};

#endif // REBIN_H
