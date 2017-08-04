#include "OscProbMatterUniform.hh"
#include <Eigen/Dense>
#include "OscillationVariables.hh"
#include "PMNSVariables.hh"
#include "ParametricLazy.hpp"

OscProbMatter::OscProbMatter(Neutrino from, Neutrino to)
    : OscProbPMNSBase(from, to), m_from(from), m_to(to)
{
    if (from.kind != to.kind) {
      throw std::runtime_error("Particle-antiparticle oscillations");
    };

    variable_(&m_L, "L");
    variable_(&m_rho, "rho");
    transformation_(this, "oscprob")
        .input("Enu")
        .output("oscprob")
        .depends(m_L)
        .depends(m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23)
        .types(Atypes::pass<0>)
        .func(&OscProbMatter::calcOscProb);

    m_param->variable_("DeltaMSq12");
    m_param->variable_("DeltaMSq13");
    m_param->variable_("DeltaMSq23");
    m_param->variable_("Theta12");
    m_param->variable_("Theta13");
    m_param->variable_("Theta23");

}

void OscProbMatter::calcOscProb(Args args, Rets rets) {
    using namespace ParametricLazyOps;
    using std::sin;
    using std::cos;
    using std::pow;
    using std::sqrt;

    double SinSq12 = pow(sin(m_param->Theta12), 2);
    double Sin12 = sin(m_param->Theta12);
    double Cos12 = cos(m_param->Theta12);
    double CosSq12 = 1-SinSq12;
    double SinSq23 = pow(sin(m_param->Theta23), 2);
    double Sin23 = sin(m_param->Theta23);
    double Cos23 = cos(m_param->Theta23);
    double CosSq23 = 1-SinSq23;
    double SinSq13 = pow(sin(m_param->Theta13), 2);
    double Sin13 = sin(m_param->Theta13);
    double Cos13 = cos(m_param->Theta23);
    double CosSq13 = 1-SinSq13;


    auto& Enu = args[0].arr;
    Eigen::ArrayXd E = Enu*1E6;
    double m_qe = -2.0/3.0*7.63E-14*0.5*m_rho.value();


    Eigen::ArrayXd qe = (m_from.kind == Neutrino::Kind::Particle ? 2 : -2)*m_qe*Enu*1E6;



    double d1 = (-m_param->DeltaMSq12.value() - m_param->Alpha.value()*m_param->DeltaMSq13.value())/3;
    double d2 = ( m_param->DeltaMSq12.value() - m_param->Alpha.value()*m_param->DeltaMSq23.value())/3;
    double d3 = m_param->Alpha.value()*(m_param->DeltaMSq13.value() + m_param->DeltaMSq23.value())/3;

    auto deltacp = (m_from.kind == Neutrino::Kind::Particle ? 1 : -1)*m_param->Delta.value();
    const int fsign = (m_to.flavor - m_from.flavor < 0 ) ? -1 : 1;


    Eigen::ArrayXd res = Eigen::ArrayXd::Zero(qe.size());

    // XXX: COPY-PASTED SymPy output, replace it with generated include

auto tw2 = (1.0L/2.0L)*pow(d1, 2) + (1.0L/2.0L)*pow(d2, 2) + (1.0L/2.0L)*pow(d3, 2) + (3.0L/4.0L)*qe.pow(2) - 3.0L/2.0L*qe*(CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3);
auto w = (1.0L/3.0L)*sqrt(3)*tw2.sqrt();
auto dv3 = d1*d2*d3 - 1.0L/4.0L*qe.pow(3) + (3.0L/4.0L)*qe.pow(2)*(CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3) - 1.0L/4.0L*qe*(6*CosSq12*CosSq13*d2*d3 + 6*CosSq13*SinSq12*d1*d3 + 6*SinSq13*d1*d2 + pow(d1, 2) + pow(d2, 2) + pow(d3, 2));
auto tcos = ((1.0L/3.0L)*((3.0L/2.0L)*dv3*(tw2*w).inverse()).acos()).cos();
auto Em = -w*(tcos + sqrt(-3*pow(tcos, 2) + 3));
auto Ep = 2*tcos*w;
auto E0 = -Em - Ep;
auto tmp0 = d1*d2 + d1*d3 + d2*d3 - 3.0L/4.0L*qe.square() + (3.0L/2.0L)*qe*(CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3);
auto sumxi0 = 3*Em.square() + tmp0;
auto sumxi1 = 3*E0.square() + tmp0;
auto sumxi2 = 3*Ep.square() + tmp0;
double tmp1 = CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3;
Eigen::ArrayXd tmp2 = CosSq12*CosSq13*d2*d3 + CosSq13*SinSq12*d1*d3 + SinSq13*d1*d2 + (1.0L/4.0L)*qe.square() + (1.0L/2.0L)*qe*(-CosSq12*CosSq13*d1 + CosSq12*CosSq13*d2 - CosSq13*d2 + CosSq13*d3 + d1 + d2);
Eigen::ArrayXd xi0_0 = (Em.square() - Em*qe + Em*tmp1 + tmp2)*sumxi0.inverse();
Eigen::ArrayXd xi0_1 = (E0.square() - E0*qe + E0*tmp1 + tmp2)*sumxi1.inverse();
Eigen::ArrayXd xi0_2 = (Ep.square() - Ep*qe + Ep*tmp1 + tmp2)*sumxi2.inverse();
double tmp3 = 2*Cos12*Cos23*Sin12*Sin13*Sin23*d1*cos(deltacp) - 2*Cos12*Cos23*Sin12*Sin13*Sin23*d2*cos(deltacp) + CosSq12*CosSq23*d2 + CosSq12*SinSq13*SinSq23*d1 + CosSq13*SinSq23*d3 + CosSq23*SinSq12*d1 + SinSq12*SinSq13*SinSq23*d2;
auto tmp4 = CosSq13*SinSq23*d1*d2 + d1*d3*(-2*Cos12*Cos23*Sin12*Sin13*Sin23*cos(deltacp) + SinSq12*SinSq13*SinSq23 + SinSq12*SinSq23 - SinSq12 - SinSq23 + 1) + d2*d3*(2*Cos12*Cos23*Sin12*Sin13*Sin23*cos(deltacp) - SinSq12*SinSq13*SinSq23 - SinSq12*SinSq23 + SinSq12 + SinSq13*SinSq23) - 1.0L/2.0L*qe.square() - 1.0L/2.0L*qe*(-4*Cos12*Cos23*Sin12*Sin13*Sin23*d1*cos(deltacp) + 4*Cos12*Cos23*Sin12*Sin13*Sin23*d2*cos(deltacp) - CosSq12*CosSq13*d1 + 2*CosSq12*CosSq23*SinSq13*d1 + 2*CosSq12*SinSq23*d2 + 2*CosSq13*CosSq23*d3 - CosSq13*SinSq12*d2 + 2*CosSq23*SinSq12*SinSq13*d2 + 2*SinSq12*SinSq23*d1 - SinSq13*d3);
Eigen::ArrayXd xi1_0 = (Em.square() + (1.0L/2.0L)*Em*qe + Em*tmp3 + tmp4)*sumxi0.inverse();
Eigen::ArrayXd xi1_1 = (E0.square() + (1.0L/2.0L)*E0*qe + E0*tmp3 + tmp4)*sumxi1.inverse();
Eigen::ArrayXd xi1_2 = (pow(Ep, 2) + (1.0L/2.0L)*Ep*qe + Ep*tmp3 + tmp4)*sumxi2.inverse();
if (m_from.flavor == m_to.flavor) {
  if (m_from.flavor == Neutrino::Flavor::Muon) {
      auto pconst = xi1_0.square() + xi1_1.square() + xi1_2.square();
      auto p01 = 2*((1.0L/2.0L)*m_L.value()*(E0 - Em)/E).cos()*(xi1_0*xi1_1).abs();
      auto p02 = 2*((1.0L/2.0L)*m_L.value()*(Em - Ep)/E).cos()*(xi1_0*xi1_2).abs();
      auto p12 = 2*((1.0L/2.0L)*m_L.value()*(E0 - Ep)/E).cos()*(xi1_1*xi1_2).abs();
      auto poscill = p01 + p02 + p12;
      res = pconst + poscill;
  }
  if (m_from.flavor == Neutrino::Flavor::Electron) {
      auto pconst = pow(xi0_0, 2) + pow(xi0_1, 2) + pow(xi0_2, 2);
      auto p01 = 2*((1.0L/2.0L)*m_L.value()*(E0 - Em)/E).cos()*(xi0_0*xi0_1).abs();
      auto p02 = 2*((1.0L/2.0L)*m_L.value()*(Em - Ep)/E).cos()*(xi0_0*xi0_2).abs();
      auto p12 = 2*((1.0L/2.0L)*m_L.value()*(E0 - Ep)/E).cos()*(xi0_1*xi0_2).abs();
      auto poscill = p01 + p02 + p12;
      res = pconst + poscill;
  }
} else {
    auto pconst = xi0_0*xi1_0 + xi0_1*xi1_1 + xi0_2*xi1_2;
    double tmp5 = Cos13*(-Cos12*Cos23*Sin12*d1 + Cos12*Cos23*Sin12*d2 - CosSq12*Sin13*Sin23*d1*cos(deltacp) - Sin13*Sin23*SinSq12*d2*cos(deltacp) + Sin13*Sin23*d3*cos(deltacp));
    double tmp6 = Cos13*(-Cos12*d2*d3*(Cos12*Sin13*Sin23*cos(deltacp) + Cos23*Sin12) + Sin12*d1*d3*(Cos12*Cos23 - Sin12*Sin13*Sin23*cos(deltacp)) + Sin13*Sin23*d1*d2*cos(deltacp));
    double tmp7 = Cos13*Sin13*Sin23*(CosSq12*d1 + SinSq12*d2 - d3)*sin(deltacp);
    double tmp8 = Cos13*Sin13*Sin23*(CosSq12*d2*d3 + SinSq12*d1*d3 - d1*d2)*sin(deltacp);
    Eigen::ArrayXd B0re = Em*tmp5 - 1.0L/2.0L*qe*tmp5 + tmp6;
    Eigen::ArrayXd B0im = Em*tmp7 - 1.0L/2.0L*qe*tmp7 + tmp8;
    Eigen::ArrayXd B0abs = sqrt(B0im.square() + B0re.square());
    Eigen::ArrayXd cosPsi0 = Eigen::ArrayXd::Zero(qe.size());
    Eigen::ArrayXd sinPsi0 = Eigen::ArrayXd::Zero(qe.size());
    for (auto i=0; i<B0abs.size(); ++i) {
          if (fabs(B0abs(i)) > 1E-10) {
            cosPsi0(i) = B0re(i)/B0abs(i);
            sinPsi0(i) = B0im(i)/B0abs(i);
          }
    };

    Eigen::ArrayXd B1re = E0*tmp5 - 1.0L/2.0L*qe*tmp5 + tmp6;
    Eigen::ArrayXd B1im = E0*tmp7 - 1.0L/2.0L*qe*tmp7 + tmp8;
    Eigen::ArrayXd B1abs = sqrt(B1im.square() + B1re.square());
    Eigen::ArrayXd cosPsi1 = Eigen::ArrayXd::Zero(qe.size());
    Eigen::ArrayXd sinPsi1 = Eigen::ArrayXd::Zero(qe.size());
    for (auto i=0; i<B1abs.size(); ++i) {
          if (fabs(B1abs(i)) > 1E-10) {
            cosPsi1(i) = B1re(i)/B1abs(i);
            sinPsi1(i) = B1im(i)/B1abs(i);
          }
    };
    Eigen::ArrayXd B2re = Ep*tmp5 - 1.0L/2.0L*qe*tmp5 + tmp6;
    Eigen::ArrayXd B2im = Ep*tmp7 - 1.0L/2.0L*qe*tmp7 + tmp8;
    Eigen::ArrayXd B2abs = sqrt(B2im.square() + B2re.square());
    Eigen::ArrayXd cosPsi2 = Eigen::ArrayXd::Zero(qe.size());
    Eigen::ArrayXd sinPsi2 = Eigen::ArrayXd::Zero(qe.size());
    for (auto i=0; i<B2abs.size(); ++i) {
          if (fabs(B2abs(i)) > 1E-10) {
            cosPsi2(i) = B2re(i)/B2abs(i);
            sinPsi2(i) = B2im(i)/B2abs(i);
          }
    }

    Eigen::ArrayXd p01 = Eigen::ArrayXd::Zero(qe.size());
    Eigen::ArrayXd E_tmp1 = (E0 - Em)/E;
    Eigen::ArrayXd phase01 = Eigen::ArrayXd::Zero(qe.size());
    for (auto i=0; i<B2abs.size(); ++i) {
      if ((fabs(B0abs(i)) > 1E-10) && (fabs(B1abs(i)) > 1E-10)) {
        phase01(i) = -1.0L/2.0L*m_L.value()*E_tmp1(i);
      }
    };
    p01 = -2*sqrt(xi0_0*xi0_1*xi1_0*xi1_1)*(fsign*(cosPsi0*sinPsi1 - cosPsi1*sinPsi0)*sin(phase01) + (cosPsi0*cosPsi1 + sinPsi0*sinPsi1)*cos(phase01));

    Eigen::ArrayXd p02 = Eigen::ArrayXd::Zero(qe.size());
    Eigen::ArrayXd E_tmp2 = (Em - Ep)/E;
    Eigen::ArrayXd phase02 = Eigen::ArrayXd::Zero(qe.size());
    for (auto i=0; i<B2abs.size(); ++i) {
      if ((fabs(B0abs(i)) > 1E-10) && (fabs(B2abs(i)) > 1E-10)) {
        phase02(i) = (1.0L/2.0L)*m_L.value()*E_tmp2(i);
      }
    };
    p02 = 2*sqrt(xi0_0*xi0_2*xi1_0*xi1_2)*(fsign*(cosPsi0*sinPsi2 - cosPsi2*sinPsi0)*sin(phase02) + (cosPsi0*cosPsi2 + sinPsi0*sinPsi2)*cos(phase02));

    Eigen::ArrayXd p12 = Eigen::ArrayXd::Zero(qe.size());
    Eigen::ArrayXd E_tmp12 = (E0 - Ep)/E;
    Eigen::ArrayXd phase12 = Eigen::ArrayXd::Zero(qe.size());
    for (auto i=0; i<B2abs.size(); ++i) {
      if ((fabs(B1abs(i)) > 1E-10) && (fabs(B2abs(i)) > 1E-10)) {
          phase12(i) = (1.0L/2.0L)*m_L.value()*E_tmp12(i);
      }
    }
    p12 = -2*sqrt(xi0_1*xi0_2*xi1_1*xi1_2)*(fsign*(cosPsi1*sinPsi2 - cosPsi2*sinPsi1)*sin(phase12) + (cosPsi1*cosPsi2 + sinPsi1*sinPsi2)*cos(phase12));

  auto poscill = p01 + p02 + p12;
  res = pconst + poscill;
}

    // End of the COPY-PASTE

  rets[0].arr = res;
} 



