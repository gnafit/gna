#include <boost/math/constants/constants.hpp>
#include "EnergyResolution.hh"
#include "TypesFunctions.hh"
#include <fmt/format.h>
#include <string.h>

constexpr double pi = boost::math::constants::pi<double>();


EnergyResolution::EnergyResolution(bool propagate_matrix) :
EnergyResolution({"Eres_a" , "Eres_b" , "Eres_c"}, propagate_matrix){

}

EnergyResolution::EnergyResolution(const std::vector<std::string>& pars, bool propagate_matrix) :
HistSmearSparse(propagate_matrix)
{
  if(pars.size()!=3u){
    throw std::runtime_error("Energy resolution should have exactly 3 parameters");
  }
  variable_(&m_a, pars[0]);
  variable_(&m_b, pars[1]);
  variable_(&m_c, pars[2]);

  transformation_("matrix")
      .input("Edges", /*inactive*/true)
      .output("FakeMatrix")
      .types(TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>, TypesFunctions::toMatrix<0,0,0>)
      .types(&EnergyResolution::getEdges)
      .func(&EnergyResolution::calcMatrix);

  add_transformation();
  add_input();
  set_open_input();
}

void EnergyResolution::getEdges(TypesFunctionArgs& fargs) {
  m_edges = fargs.args[0].edges.data();
}

double EnergyResolution::relativeSigma(double Etrue) const noexcept {
  return sqrt(pow(m_a, 2)+ pow(m_b, 2)/Etrue + pow(m_c/Etrue, 2));
}

double EnergyResolution::resolution(double Etrue, double Erec) const noexcept {
  static const double twopisqr = std::sqrt(2*pi);
  const double sigma = Etrue * relativeSigma(Etrue);
  const double reldiff = (Etrue - Erec)/sigma;

  return std::exp(-0.5*pow(reldiff, 2))/(twopisqr*sigma);
}

void EnergyResolution::calcMatrix(FunctionArgs& fargs) {
  m_sparse_cache.setZero();

  auto& ret = fargs.rets[0];
  auto* edges = m_edges;
  auto bins = ret.type.shape[0];
  m_sparse_cache.resize(bins, bins);

  /* fill the cache matrix with probalilities for number of events to leak to other bins */
  /* colums corressponds to reconstrucred energy and rows to true energy */
  auto bin_center = [edges](size_t index){ return (edges[index+1] + edges[index])/2; };
  for (size_t etrue = 0; etrue < bins; ++etrue) {
    double Etrue = bin_center(etrue);
    double dEtrue = edges[etrue+1] - edges[etrue];

    bool right_edge_reached{false};
    /* precalculating probabilities for events in given bin to leak to
     * neighbor bins  */
    for (size_t erec = 0; erec < bins; ++erec) {
      double Erec = bin_center(erec);
      double rEvents = dEtrue*resolution(Etrue, Erec);

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
