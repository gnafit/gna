#include <boost/math/constants/constants.hpp>
#include "EnergyResolution.hh"
#include <boost/format.hpp>
#include <string.h>
using boost::format;

constexpr double pi = boost::math::constants::pi<double>();

EnergyResolution::EnergyResolution( bool single ) : m_single(single) {
  variable_(&m_a, "Eres_a");
  variable_(&m_b, "Eres_b");
  variable_(&m_c, "Eres_c");
  callback_([this] { fillCache(); });

  if (single) {
    add();
  }
}

void EnergyResolution::add(){
  int index=static_cast<int>(transformations.size());
  std::string label="smear";
  if(!m_single){
    label=(format("smear_%1%")%index).str();
  }
  transformation_(this, label)
    .input("Nvis")
    .output("Nrec")
    .types(Atypes::pass<0>,Atypes::ifHist<0>,
           [index](EnergyResolution *obj, Atypes args, Rtypes /*rets*/) {
             if(index==0){
               obj->m_datatype = args[0];
               obj->fillCache();
             }
             else{
               if( args[0]!=obj->m_datatype ) {
                 throw std::runtime_error("Inconsistent histogram in EnergyResolution");
               }
             }
           })
  .func(&EnergyResolution::calcSmear);
}

void EnergyResolution::add(SingleOutput& hist){
  if( m_single ){
    throw std::runtime_error("May have only single energy resolution transformation in this class instance");
  }
  add();
  transformations.back().inputs[0]( hist.single() );
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

void EnergyResolution::fillCache() {
  m_size = m_datatype.hist().bins();
  if (m_size == 0) {
    return;
  }
  m_sparse_cache.resize(m_size, m_size);

  /* fill the cache matrix with probalilities for number of events to leak to other bins */
  /* colums corressponds to reconstrucred energy and rows to true energy */
  auto bin_center = [&](size_t index){ return (m_datatype.edges[index+1] + m_datatype.edges[index])/2; };
  for (size_t etrue = 0; etrue < m_size; ++etrue) {
    double Etrue = bin_center(etrue);
    double dEtrue = m_datatype.edges[etrue+1] - m_datatype.edges[etrue];

    bool right_edge_reached{false};
    /* precalculating probabilities for events in given bin to leak to
     * neighbor bins  */
    for (size_t erec = 0; erec < m_size; ++erec) {
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
}

/* Apply precalculated cache and actually smear */
void EnergyResolution::calcSmear(Args args, Rets rets) {
  rets[0].x = m_sparse_cache * args[0].vec;
}



