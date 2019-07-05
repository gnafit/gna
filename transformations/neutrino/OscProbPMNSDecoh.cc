#include <TMath.h>

#include "OscProbPMNSDecoh.hh"
#include "OscillationVariables.hh"
#include "PMNSVariables.hh"
#include "Units.hh"
#include "TypesFunctions.hh"

using namespace Eigen;
using namespace NeutrinoUnits;

using namespace GNA::GNAObjectTemplates;

template<typename FloatType>
GNA::GNAObjectTemplates::OscProbPMNSDecohT<FloatType>::OscProbPMNSDecohT(Neutrino from, Neutrino to)
  : OscProbPMNSBaseT<FloatType>(from, to)
{
  this->variable_(&m_L, "L");
  this->variable_(&m_sigma, "SigmaDecohRel");

  this->transformation_("comp12")
    .input("Enu")
    .output("comp12")
    .output("compCP12")
    .depends(m_L, m_sigma, this->m_param->DeltaMSq12)
    .func(&OscProbPMNSDecohT<FloatType>::calcComponent<1,2>);
  this->transformation_("comp13")
    .input("Enu")
    .output("comp13")
    .output("compCP13")
    .depends(m_L, m_sigma, this->m_param->DeltaMSq13)
    .func(&OscProbPMNSDecohT<FloatType>::calcComponent<1,3>);
  this->transformation_("comp23")
    .input("Enu")
    .output("comp23")
    .output("compCP23")
    .depends(m_L, m_sigma, this->m_param->DeltaMSq23)
    .func(&OscProbPMNSDecohT<FloatType>::calcComponent<2,3>);
   auto probsum = this->transformation_("probsum")
    .input("comp12")
    .input("comp13")
    .input("comp23")
    .output("probsum")
    .types(TypesFunctions::pass<0>)
    .func(&OscProbPMNSDecohT<FloatType>::calcSum);
  if (from.flavor != to.flavor) {
    probsum.input("compCP12");
    probsum.input("compCP13");
    probsum.input("compCP23");
  } else {
    probsum.input("comp0");
  }
}

/* // Was replaced to header as it has template params
template <int I, int J>
void OscProbPMNSDecoh::calcComponent(FunctionArgs fargs) {
  auto& rets=fargs.rets;
  auto &Enu = fargs.args[0].x;
  ArrayXd phi_st = DeltaMSq<I,J>()*eV2*m_L*km/2.0*Enu.inverse()/MeV;
  ArrayXd Lcoh = TMath::Sqrt2()*2*Enu*MeV/DeltaMSq<I,J>()/eV2/m_sigma;
  ArrayXd Ld   = Lcoh/m_sigma/(TMath::Sqrt2()*2);
  ArrayXd rd = m_L*km/Ld;
  ArrayXd rd2 = 1.e0+rd.square();
  ArrayXd rc2 = (m_L*km/Lcoh).square();
  ArrayXd phi_dd = Ld.unaryExpr([this](double y) { return std::atan2(m_L*km, y); });
  ArrayXd phi_d = -1.e0/rd2*rc2*rd+0.5*phi_dd;
  ArrayXd D2 = 0.5*(DeltaMSq<I,J>()*eV2/4/m_sigma/Enu.square()/MeV/MeV).square();
  ArrayXd F = 1.0/rd2.sqrt().sqrt()*exp(-1./rd2*rc2-D2);
  rets[0].x = 1.e0 - F*cos(phi_st+phi_d);
  if (m_alpha != m_beta) {
    rets[1].x = F*sin(phi_st+phi_d);
  }
}
*/

template<typename FloatType>
void GNA::GNAObjectTemplates::OscProbPMNSDecohT<FloatType>::calcSum(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0].x;
  ret = -(this->template weight<1,2>())*(FloatType)3.0*args[0].x;
  ret+= -2.0*(this->template weight<1,3>())*args[1].x;
  ret+= -2.0*(this->template weight<2,3>())*args[2].x;
  if (this->m_alpha == this->m_beta) {
    ret += args[3].x;
  } else {
    ret += 8.0*this->weightCP()*(args[3].x-args[4].x+args[5].x);
  }
}
