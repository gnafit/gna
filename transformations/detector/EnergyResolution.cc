#include <boost/math/constants/constants.hpp>
#include "EnergyResolution.hh"
#include "TypesFunctions.hh"
#include <fmt/format.h>
#include <string.h>

constexpr double pi = boost::math::constants::pi<double>();

EnergyResolution::EnergyResolution(bool single, bool propagate_matrix) :
m_propagate_matrix(propagate_matrix),
m_single(single)
{
  variable_(&m_a, "Eres_a");
  variable_(&m_b, "Eres_b");
  variable_(&m_c, "Eres_c");

  transformation_("matrix")
      .input("Edges")
      .output("FakeMatrix")
      .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>, TypesFunctions::edgesToMatrix<0,0,0>)
      .func(&EnergyResolution::calcMatrix);

  if (single) {
    add();
  }
}

TransformationDescriptor EnergyResolution::add(){
  int index=static_cast<int>(transformations.size());
  std::string label="smear";
  if(!m_single){
    label = fmt::format("smear_{0}", index);
  }
  auto init=transformation_(label)
    .input("Nvis")
    .input("FakeMatrix")
    .output("Nrec")
    .dont_subscribe()
    .types(TypesFunctions::pass<0>, TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>)
    .types(TypesFunctions::if2d<1>, TypesFunctions::ifSquare<1>,
           [](TypesFunctionArgs& fargs){
           auto& args = fargs.args;
           auto& vec = args[0];
           auto& mat = args[1];
           if( vec.shape[0]!=mat.shape[0] ) {
             throw args.error(vec, "Inputs are not multiplicable");
           }
           })
    .func(&EnergyResolution::calcSmear);

  t_[label].inputs()[1].connect( t_["matrix"].outputs()[0] );
  return transformations.back();
}

TransformationDescriptor EnergyResolution::add(SingleOutput& hist){
  if( m_single ){
    throw std::runtime_error("May have only single energy resolution transformation in this class instance");
  }
  add();
  transformations.back().inputs[0]( hist.single() );
  return transformations.back();
}

double EnergyResolution::relativeSigma(double Etrue) const noexcept{
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

  auto& arg = fargs.args[0];
  auto* edges = arg.buffer;
  auto bins = arg.type.shape[0]-1;
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

/* Apply precalculated cache and actually smear */
void EnergyResolution::calcSmear(FunctionArgs& fargs) {
  auto& args=fargs.args;
  args[1]; // needed to trigger the matrix update
  fargs.rets[0].x = m_sparse_cache * fargs.args[0].vec;
}



