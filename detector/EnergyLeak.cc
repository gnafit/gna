#include <boost/math/constants/constants.hpp>
#include "EnergyLeak.hh"

constexpr double pi = boost::math::constants::pi<double>();

EnergyLeak::EnergyLeak(size_t n, double* mat_column_major) :
m_size(n), m_matrix(Eigen::Map<Eigen::MatrixXd>(mat_column_major, n, n)) {
  //callback_([this] { fillCache(); });

  transformation_(this, "smear")
      .input("Ntrue")
      .output("Nvis")
      .types(Atypes::pass<0>,
         [](EnergyLeak *obj, Atypes args, Rtypes /*rets*/) {
           obj->m_datatype = args[0];
           obj->fillCache();
         })
       .func(&EnergyLeak::calcSmear);
}

void EnergyLeak::fillCache() {
}

void EnergyLeak::calcSmear(Args args, Rets rets) {
  rets[0].x = m_matrix.triangularView<Eigen::Lower>() * args[0].vec;
}
