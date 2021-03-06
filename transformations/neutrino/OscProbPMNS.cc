#include <array>

#include <TMath.h>

#include "OscProbPMNS.hh"
#include <Eigen/Dense>

#include "OscillationVariables.hh"
#include "PMNSVariables.hh"
#include "TypesFunctions.hh"
#include "Units.hh"

#include "TypesFunctions.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

#ifdef GNA_CUDA_SUPPORT
#include "cuOscProbPMNS.hh"
#endif

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
    m_pmns->V[m_alpha][I-1].complex()*
    m_pmns->V[m_beta][J-1].complex()*
    std::conj(m_pmns->V[m_alpha][J-1].complex())*
    std::conj(m_pmns->V[m_beta][I-1].complex())
    );
}

double OscProbPMNSBase::weightCP() const {
  return m_lepton_charge*std::imag(
    m_pmns->V[m_alpha][0].complex()*
    m_pmns->V[m_beta][1].complex()*
    std::conj(m_pmns->V[m_alpha][1].complex())*
    std::conj(m_pmns->V[m_beta][0].complex())
    );
}


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


template<typename FloatType>
GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::OscProbPMNST(Neutrino from, Neutrino to, std::string l_name)
  : OscProbPMNSBase(from, to)
{
  variable_(&m_L, l_name);
  this->transformation_("comp12")
    .input("Enu")
    .output("comp12")
    .depends(m_L, m_param->DeltaMSq12)
    .types(new PassTypeT<FloatType>(0, {0,-1}))
    .func(&OscProbPMNST<FloatType>::calcComponent<1,2>)
#ifdef GNA_CUDA_SUPPORT
    .func("gpu", &OscProbPMNST<FloatType>::gpuCalcComponent<1,2>, DataLocation::Device)
    .storage("gpu", [](StorageTypesFunctionArgs& fargs){
      fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
    })
#endif
  ;
  this->transformation_("comp13")
    .input("Enu")
    .output("comp13")
    .depends(m_L, m_param->DeltaMSq13)
    .types(new PassTypeT<FloatType>(0, {0,-1}))
    .func(&OscProbPMNST<FloatType>::calcComponent<1,3>)
#ifdef GNA_CUDA_SUPPORT
    .func("gpu", &OscProbPMNST<FloatType>::gpuCalcComponent<1,3>, DataLocation::Device)
    .storage("gpu", [](StorageTypesFunctionArgs& fargs){
      fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
    })
#endif
  ;
  this->transformation_("comp23")
    .input("Enu")
    .output("comp23")
    .depends(m_L, m_param->DeltaMSq23)
    .types(new PassTypeT<FloatType>(0, {0,-1}))
    .func(&OscProbPMNST<FloatType>::calcComponent<2,3>)
#ifdef GNA_CUDA_SUPPORT
    .func("gpu", &OscProbPMNST<FloatType>::gpuCalcComponent<2,3>, DataLocation::Device)
    .storage("gpu", [](StorageTypesFunctionArgs& fargs){
      fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
    })
#endif
  ;
  if (m_alpha != m_beta) {
    this->transformation_("compCP")
      .input("Enu")
      .output("compCP")
      .depends(m_L)
      .depends(m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23)
      .types(new PassTypeT<FloatType>(0, {0,-1}))
      .func(&OscProbPMNST<FloatType>::calcComponentCP)
#ifdef GNA_CUDA_SUPPORT
      .func("gpu", &OscProbPMNST<FloatType>::gpuCalcComponentCP, DataLocation::Device)
      .storage("gpu", [](StorageTypesFunctionArgs& fargs){
        fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
      })
#endif
      ;
  }
  auto probsum = this->transformation_("probsum")
    .input("comp12")
    .input("comp13")
    .input("comp23")
    .input("comp0")
    .output("probsum")
    .types(new PassTypeT<FloatType>(0, {0,-1}))
    .func(&OscProbPMNST<FloatType>::calcSum)
#ifdef GNA_CUDA_SUPPORT
    .func("gpu", &OscProbPMNST<FloatType>::gpuCalcSum, DataLocation::Device)
#endif
    ;
  if (from.flavor != to.flavor) {
    probsum.input("compCP");
  }

  this->transformation_("full_osc_prob")
      .input("Enu")
      .output("oscprob")
      .depends(m_L, m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23)
      .types(new PassTypeT<FloatType>(0, {0,-1}))
      .func(&OscProbPMNST<FloatType>::calcFullProb);
}

template<typename FloatType>
void GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::calcFullProb(FunctionArgs& fargs) {
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


template<typename FloatType>
template <int I, int J>
void GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::calcComponent(FunctionArgs& fargs) {
  auto &Enu = fargs.args[0].x;
  fargs.rets[0].x = cos((DeltaMSq<I,J>()*oscprobArgumentFactor*m_L*0.5)*Enu.inverse());
}

#ifdef GNA_CUDA_SUPPORT
template<typename FloatType>
template < int I, int J>
void GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::gpuCalcComponent(FunctionArgs& fargs) {
  fargs.args.touch();
  auto& gpuargs=fargs.gpu;
  gpuargs->provideSignatureDevice();
  cuCalcComponent_modecos<double>(gpuargs->args, gpuargs->rets, gpuargs->ints, gpuargs->vars,
		                  fargs.args[0].arr.size(), gpuargs->nargs, oscprobArgumentFactor, DeltaMSq<I,J>(), m_L);
}
#endif

template<typename FloatType>
void GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::calcComponentCP(FunctionArgs& fargs) {
  auto& ret=fargs.rets[0].x;
  auto &Enu = fargs.args[0].x;
  ArrayXd tmp = (oscprobArgumentFactor*m_L*0.25)*Enu.inverse();
  ret = sin(DeltaMSq<1,2>()*tmp);
  ret*= sin(DeltaMSq<1,3>()*tmp);
  ret*= sin(DeltaMSq<2,3>()*tmp);
}

#ifdef GNA_CUDA_SUPPORT
template<typename FloatType>
void GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::gpuCalcComponentCP(FunctionArgs& fargs) {
  fargs.args.touch();
  auto& gpuargs=fargs.gpu;
  gpuargs->provideSignatureDevice();
  cuCalcComponentCP<double>(gpuargs->args, gpuargs->rets, gpuargs->ints, gpuargs->vars, m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23,
			    fargs.args[0].arr.size(), gpuargs->nargs, oscprobArgumentFactor, m_L);
}
#endif

template<typename FloatType>
void GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::calcSum(FunctionArgs& fargs) {
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
#ifdef GNA_CUDA_SUPPORT
template<typename FloatType>
void GNA::GNAObjectTemplates::OscProbPMNST<FloatType>::gpuCalcSum(FunctionArgs& fargs) {
  fargs.args.touch();
  auto& gpuargs=fargs.gpu;
  cuCalcSum(gpuargs->args, gpuargs->rets, weight<1,2>(), weight<1,3>(), weight<2,3>() ,weightCP(), (m_alpha == m_beta), fargs.args[0].arr.size());
}
#endif

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
  auto& svalues=m_weights.values();
  double s2 = svalues[0];
  double s3 = svalues[1];
  double s4 = svalues[2];
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
template class GNA::GNAObjectTemplates::OscProbPMNST<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  //template class GNA::GNAObjectTemplates::OscProbPMNST<float>;
#endif
