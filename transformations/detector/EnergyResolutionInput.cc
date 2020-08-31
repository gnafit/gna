#include <boost/math/constants/constants.hpp>
#include "EnergyResolutionInput.hh"
#include "TypeClasses.hh"
#include "TypesFunctions.hh"
#include <fmt/format.h>
#include <string.h>

constexpr double pi = boost::math::constants::pi<double>();

using namespace TypeClasses;

EnergyResolutionInput::EnergyResolutionInput(bool propagate_matrix) :
HistSmearSparse(propagate_matrix)
{
  transformation_("matrix")
      .input("Edges", /*inactive*/true)     // Input bin edges [N]
      .input("RelSigma")                    // Relative Sigma value for each bin center [N-1]
      .output("FakeMatrix")
      .types(new CheckKindT<double>(DataKind::Hist, 0), new CheckKindT<double>(DataKind::Points, 1))
      .types(new CheckNdimT<double>(1), new CheckSameTypesT<double>({0,1}, "shape"))
      .types(TypesFunctions::toMatrix<0,0,0>)
      .func(&EnergyResolutionInput::calcMatrix);

  add_transformation();
  add_input();
  set_open_input();
}

double EnergyResolutionInput::resolution(double Etrue, double Erec, double RelSigma) const noexcept {
  static const double twopisqr = std::sqrt(2*pi);
  const double sigma = Etrue * RelSigma;
  const double reldiff = (Etrue - Erec)/sigma;

  return std::exp(-0.5*pow(reldiff, 2))/(twopisqr*sigma);
}

void EnergyResolutionInput::calcMatrix(FunctionArgs& fargs) {
  m_sparse_cache.setZero();

  auto& args = fargs.args;
  auto* edges = args[0].type.edges.data();
  auto& relsigmas = args[1].x;
  auto& ret = fargs.rets[0];
  size_t nbins = ret.type.shape[0];

  m_sparse_cache.resize(nbins, nbins);

  /* fill the cache matrix with probalilities for number of events to leak to other bins */
  /* colums corressponds to reconstrucred energy and rows to true energy */
  auto bin_center = [edges](size_t index){ return (edges[index+1] + edges[index])/2; };
  for (size_t etrue = 0; etrue < nbins; ++etrue) {
    auto Etrue   = bin_center(etrue);
    auto dEtrue  = edges[etrue+1] - edges[etrue];
    auto relsigma=relsigmas[etrue];

    bool right_edge_reached{false};
    /* precalculating probabilities for events in given bin to leak to
     * neighbor bins  */
    for (size_t erec = 0; erec < nbins; ++erec) {
      auto Erec = bin_center(erec);
      auto rEvents = dEtrue*resolution(Etrue, Erec, relsigma);

      if (rEvents < 1E-10) {
        if (right_edge_reached) {
           break;
        }
        continue;
      }
      m_sparse_cache.insert(erec, etrue) = rEvents;
      if (!right_edge_reached) {
        right_edge_reached = true;
      }
    }
  }
  m_sparse_cache.makeCompressed();

  if ( m_propagate_matrix )
    fargs.rets[0].mat = m_sparse_cache;
}
