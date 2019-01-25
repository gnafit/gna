#include <array>

#include <TMath.h>

#include "OscProbPMNS.hh"
#include <Eigen/Dense>

#include "OscillationVariables.hh"
#include "PMNSVariables.hh"
#include "TypesFunctions.hh"
#include "Units.hh"

#ifdef GNA_CUDA_SUPPORT 
//#include "extra/GNAcuOscProbFull.h"
//#include "extra/GNAcuOscProbMem.hh"
#endif

#include <chrono>
#include <ctime>


using namespace Eigen;
using NeutrinoUnits::oscprobArgumentFactor;

OscProbPMNSBase::OscProbPMNSBase(Neutrino from, Neutrino to)
  : m_param(new OscillationVariables(this)), m_pmns(new PMNSVariables(this))
{
  if (from.kind != to.kind) {
    throw std::runtime_error("particle-antiparticle oscillations");
  }
  m_alpha = from.flavor;
  m_beta = to.flavor;
  m_lepton_charge = from.leptonCharge();

  for (size_t i = 0; i < m_pmns->Nnu; ++i) {
    m_pmns->variable_(&m_pmns->V[m_alpha][i]);
    m_pmns->variable_(&m_pmns->V[m_beta][i]);
  }
  m_param->variable_("DeltaMSq12");
  m_param->variable_("DeltaMSq13");
  m_param->variable_("DeltaMSq23");
}

template <>
double OscProbPMNSBase::DeltaMSq<1,2>() const { return m_param->DeltaMSq12; }

template <>
double OscProbPMNSBase::DeltaMSq<1,3>() const { return m_param->DeltaMSq13; }

template <>
double OscProbPMNSBase::DeltaMSq<2,3>() const { return m_param->DeltaMSq23; }

template <int I, int J>
double OscProbPMNSBase::weight() const {
  return std::real(
    m_pmns->V[m_alpha][I-1].value()*
    m_pmns->V[m_beta][J-1].value()*
    std::conj(m_pmns->V[m_alpha][J-1].value())*
    std::conj(m_pmns->V[m_beta][I-1].value())
    );
}

double OscProbPMNSBase::weightCP() const {
  return m_lepton_charge*std::imag(
    m_pmns->V[m_alpha][0].value()*
    m_pmns->V[m_beta][1].value()*
    std::conj(m_pmns->V[m_alpha][1].value())*
    std::conj(m_pmns->V[m_beta][0].value())
    );
}

//OscProbPMNS::OscProbPMNSWeights(Neutrino from, Neutrino to)
  //: OscProbPMNSBase(from, to)
//{
  //std::vector<changeable> deps;
  //deps.reserve(4+static_cast<int);

    //.input("weight12")
    //.input("weight13")
    //.input("weight23")
    //.input("weight0")

  //for (size_t i = 0; i < varnames.size(); ++i) {
    //variable_(&m_vars[i], varnames[i]);
    //deps.push_back(m_vars[i]);
  //}
  //m_sum = evaluable_<double>(sumname, [this]() {
      //double res = m_vars[0];
      //for (size_t i = 1; i < m_vars.size(); ++i) {
          //res+=m_vars[i];
      //}
      //return res;
    //}, deps);

//}

OscProbAveraged::OscProbAveraged(Neutrino from, Neutrino to):
    OscProbPMNSBase(from, to)
{
  transformation_("average_oscillations")
        .input("flux")
        .output("flux_averaged_osc")
        .types(TypesFunctions::pass<0>)
        .func(&OscProbAveraged::CalcAverage);
}

void OscProbAveraged::CalcAverage(FunctionArgs fargs) {
    double aver_weight = 1.0 - 2.0*(weight<1,2>() + weight<1,3>() + weight<2,3>());
    fargs.rets[0].x = aver_weight * fargs.args[0].x;
}


OscProbPMNS::OscProbPMNS(Neutrino from, Neutrino to, std::string l_name)
  : OscProbPMNSBase(from, to)
{
  variable_(&m_L, l_name);
  transformation_("comp12")
    .input("Enu")
    .output("comp12")
    .depends(m_L, m_param->DeltaMSq12)
    .func(&OscProbPMNS::calcComponent<1,2>);
  transformation_("comp13")
    .input("Enu")
    .output("comp13")
    .depends(m_L, m_param->DeltaMSq13)
    .func(&OscProbPMNS::calcComponent<1,3>);
  transformation_("comp23")
    .input("Enu")
    .output("comp23")
    .depends(m_L, m_param->DeltaMSq23)
    .func(&OscProbPMNS::calcComponent<2,3>);
  if (m_alpha != m_beta) {
    transformation_("compCP")
      .input("Enu")
      .output("compCP")
      .depends(m_L)
      .depends(m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23)
      .func(&OscProbPMNS::calcComponentCP);
  }
  auto probsum = transformation_("probsum")
    .input("comp12")
    .input("comp13")
    .input("comp23")
    .input("comp0")
    .output("probsum")
    .types(TypesFunctions::pass<0>)
    .func(&OscProbPMNS::calcSum);
  if (from.flavor != to.flavor) {
    probsum.input("compCP");
  }

  auto full_formula = transformation_("full_osc_prob")
      .input("Enu")
      .output("oscprob")
      .depends(m_L, m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23)
      .types(TypesFunctions::pass<0>)
      .func(&OscProbPMNS::calcFullProb);
/*
#ifdef GNA_CUDA_SUPPORT
  auto full_formula_gpu = transformation_( "full_osc_prob_gpu")
      .input("Enu")
      .output("oscprob")
      .depends(m_L, m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23)
      .types(TypesFunctions::pass<0>)
      .func(&OscProbPMNS::calcFullProbGpu);
#endif
*/

}

void OscProbPMNS::calcFullProb(FunctionArgs fargs) {
  auto& ret=fargs.rets[0].x;
  auto& Enu = fargs.args[0].x;
  ArrayXd tmp = (oscprobArgumentFactor*m_L*0.5)*Enu.inverse();
  ArrayXd comp0(Enu);
  comp0.setOnes();
  ArrayXd comp12 = cos(DeltaMSq<1,2>()*tmp);
  ArrayXd comp13 = cos(DeltaMSq<1,3>()*tmp);
  ArrayXd comp23 = cos(DeltaMSq<2,3>()*tmp);
  ArrayXd compCP(Enu);
  compCP.setZero();
  if (m_alpha != m_beta) {
    compCP  = sin(DeltaMSq<1,2>()*tmp/2.);
    compCP *= sin(DeltaMSq<1,3>()*tmp/2.);
    compCP *= sin(DeltaMSq<2,3>()*tmp/2.);
  }
  ret  = 2.0*weight<1,2>()*comp12;
  ret += 2.0*weight<1,3>()*comp13;
  ret += 2.0*weight<2,3>()*comp23;
  double coeff0 = - 2.0*(weight<1,2>() + weight<1,3>() + weight<2,3>());
  if (m_alpha == m_beta) {
    coeff0 += 1.0;
  }
  ret += coeff0*comp0;
  if (m_alpha != m_beta) {
    ret += 8.0*weightCP()*compCP;
  }
}


template <int I, int J>
void OscProbPMNS::calcComponent(FunctionArgs fargs) {
  auto &Enu = fargs.args[0].x;
  fargs.rets[0].x = cos((DeltaMSq<I,J>()*oscprobArgumentFactor*m_L*0.5)*Enu.inverse());
}

void OscProbPMNS::calcComponentCP(FunctionArgs fargs) {
  auto& ret=fargs.rets[0].x;
  auto &Enu = fargs.args[0].x;
  ArrayXd tmp = (oscprobArgumentFactor*m_L*0.25)*Enu.inverse();
  ret = sin(DeltaMSq<1,2>()*tmp);
  ret*= sin(DeltaMSq<1,3>()*tmp);
  ret*= sin(DeltaMSq<2,3>()*tmp);
}

void OscProbPMNS::calcSum(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0].x;
  auto weight12=weight<1,2>();
  auto weight13=weight<1,3>();
  auto weight23=weight<2,3>();
  ret = 2.0*weight12*args[0].x;
  ret+= 2.0*weight13*args[1].x;
  ret+= 2.0*weight23*args[2].x;
  double coeff0 = -2.0*(weight12+weight13+weight23);
  if (m_alpha == m_beta) {
    coeff0 += 1.0;
  }
  ret += coeff0*args[3].x;
  if (m_alpha != m_beta) {
    ret += 8.0*weightCP()*args[4].x;
  }
}

OscProbPMNSMult::OscProbPMNSMult(Neutrino from, Neutrino to, std::string l_name)
  : OscProbPMNSBase(from, to)
{
  if (m_alpha != m_beta) {
    throw std::runtime_error("OscProbPMNSMult is only for survivals");
  }
  variable_(&m_Lavg, l_name);
  variable_(&m_weights, "weights");

  transformation_("comp12")
    .input("Enu")
    .output("comp12")
    .depends(m_Lavg, m_param->DeltaMSq12)
    .func(&OscProbPMNSMult::calcComponent<1,2>);
  transformation_("comp13")
    .input("Enu")
    .output("comp13")
    .depends(m_Lavg, m_param->DeltaMSq13)
    .func(&OscProbPMNSMult::calcComponent<1,3>);
  transformation_("comp23")
    .input("Enu")
    .output("comp23")
    .depends(m_Lavg, m_param->DeltaMSq23)
    .func(&OscProbPMNSMult::calcComponent<2,3>);
  transformation_("probsum")
    .input("comp12")
    .input("comp13")
    .input("comp23")
    .input("comp0")
    .output("probsum")
    .types(TypesFunctions::pass<0>)
    .func(&OscProbPMNSMult::calcSum);
}

template <int I, int J>
void OscProbPMNSMult::calcComponent(FunctionArgs fargs) {
  double s2 = m_weights.value()[0];
  double s3 = m_weights.value()[1];
  double s4 = m_weights.value()[2];
  auto &Enu = fargs.args[0].x;
  ArrayXd phi = (DeltaMSq<I,J>()*oscprobArgumentFactor*m_Lavg*0.25)*Enu.inverse();
  ArrayXd phi2 = phi.square();
  ArrayXd a = 1.0 - 2.0*s2*phi2 + 2.0/3.0*s4*phi2.square();
  ArrayXd b = 1.0 - 2.0/3.0*s3*phi2;
  fargs.rets[0].x = a*cos(2.0*b*phi);
}

void OscProbPMNSMult::calcSum(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0].x;
  ret = weight<1,2>()*args[0].x;
  ret+= weight<1,3>()*args[1].x;
  ret+= weight<2,3>()*args[2].x;
  ret+= (1.0-weight<1,2>()-weight<1,3>()-weight<2,3>())*args[3].x;
}
