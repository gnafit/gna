#include <boost/math/constants/constants.hpp>
#include "EnergySmear.hh"

EnergySmear::EnergySmear(size_t n, double* mat_column_major, bool triangular) :
m_size(n), m_matrix(Eigen::Map<Eigen::MatrixXd>(mat_column_major, n, n)) {
  //callback_([this] { fillCache(); });

  transformation_(this, "smear")
      .input("Ntrue")
      .output("Nvis")
      .types(Atypes::pass<0>,
         [](EnergySmear *obj, Atypes args, Rtypes /*rets*/) {
           obj->m_datatype = args[0];
           obj->fillCache();
         })
       .func( triangular ? &EnergySmear::calcSmearTriangular : &EnergySmear::calcSmear );
}

void EnergySmear::fillCache() {
}

void EnergySmear::calcSmearTriangular(Args args, Rets rets) {
  rets[0].x = m_matrix.triangularView<Eigen::Upper>() * args[0].vec;
}

void EnergySmear::calcSmear(Args args, Rets rets) {
  rets[0].x = m_matrix * args[0].vec;
}
