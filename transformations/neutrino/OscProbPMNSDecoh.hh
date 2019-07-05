#pragma once

#include <Eigen/Dense>

#include "OscProbPMNS.hh"
#include "Neutrino.hh"

class OscillationVariables;
class PMNSVariables;

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProbPMNSDecohT: public OscProbPMNSBaseT<FloatType>,
                            public TransformationBind<OscProbPMNSDecohT<FloatType>,FloatType,FloatType> {
    //  using namespace Eigen;
    //  using namespace NeutrinoUnits;
    private:
      using BaseClass = OscProbPMNSBaseT<FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      using TransformationBind<OscProbPMNSDecohT<FloatType>,FloatType,FloatType>::transformation_;
      OscProbPMNSDecohT(Neutrino from, Neutrino to);
      void calcSum(FunctionArgs fargs);

      template <int I, int J>
      void calcComponent(FunctionArgs fargs);/* {
         auto& rets=fargs.rets;
         auto &Enu = fargs.args[0].x;
         ArrayXd phi_st = this->DeltaMSq<I,J>()*eV2*m_L*km/2.0*Enu.inverse()/MeV;
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
      } */

      template <int I, int J>
      void calcComponentCP(FunctionArgs fargs);
    protected:
      variable<FloatType> m_L;
      variable<FloatType> m_sigma;
    };
  }
}  
