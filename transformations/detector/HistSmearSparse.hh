#pragma once

#include <vector>

#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"
#include "Eigen/Sparse"
#include "HistSmearEnums.hh"

class HistSmearSparse: public GNAObjectBind1N<double>,
                       public TransformationBind<HistSmearSparse> {
public:
  HistSmearSparse(GNA::DataPropagation propagate_matrix=GNA::DataPropagation::Ignore,
                  GNA::MatrixType matrix_type=GNA::MatrixType::Square,
                  const std::string& transformationname="smear",
                  const std::string& inputname="Ntrue",
                  const std::string& outputname="Nrec");

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  TransformationDescriptor add_transformation(const std::string& name="");

protected:
  virtual std::vector<double> getOutputEdges() const noexcept { return {}; }
  Eigen::SparseMatrix<double> m_sparse_cache;
  bool m_propagate_matrix{false};
  bool m_square{true};

private:
  void types(TypesFunctionArgs& fargs);
  void calcSmear(FunctionArgs& fargs);
};
