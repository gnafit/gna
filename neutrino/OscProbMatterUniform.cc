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
    transformation_(this, "comp12")
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

};

void OscProbMatter::calcOscProb(Args args, Rets rets) {
    using namespace ParametricLazyOps;
    using std::sin;
    using std::cos;
    using std::pow;
    using std::sqrt;

    double SinSq12 = pow(sin(m_param->Theta12), 2);
    auto Sin12 = sin(m_param->Theta12);
    auto Cos12 = cos(m_param->Theta12);
    auto CosSq12 = 1-SinSq12;
    auto SinSq23 = pow(sin(m_param->Theta23), 2);
    auto Sin23 = sin(m_param->Theta23);
    auto Cos23 = cos(m_param->Theta23);
    auto CosSq23 = 1-SinSq23;
    auto SinSq13 = pow(sin(m_param->Theta13), 2);
    auto Sin13 = sin(m_param->Theta13);
    auto Cos13 = cos(m_param->Theta23);
    auto CosSq13 = 1-SinSq13;


    auto& Enu = args[0].arr;
    double m_qe = mkdep(-2.0/3.0*7.63E-14*0.5*m_rho);


    Eigen::ArrayXd qe = (m_from.kind == Neutrino::Kind::Particle ? 2 : -2)*m_qe*Enu*1E6;



    double d1 = mkdep((-m_param->DeltaMSq12 - m_param->Alpha * m_param->DeltaMSq13)/3);
    double d2 = mkdep(( m_param->DeltaMSq12 - m_param->Alpha*m_param->DeltaMSq23)/3);
    double d3 = mkdep(m_param->Alpha*(m_param->DeltaMSq13 + m_param->DeltaMSq23)/3);

    auto deltacp = mkdep((m_from.kind == Neutrino::Kind::Particle ? 1 : -1)*m_param->Delta);
    const int fsing = (m_to.flavor - m_from.flavor < 0 ) ? -1 : 1;

    auto sflavor = m_sflavor;

    double res;

    // XXX: COPY-PASTED SymPy output, replace it with generated include

Eigen::ArrayXd tw2 = (1.0L/2.0L)*pow(d1, 2) + (1.0L/2.0L)*pow(d2, 2) + (1.0L/2.0L)*pow(d3, 2) + (3.0L/4.0L)*pow(qe, 2) - 3.0L/2.0L*qe*(CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3);
auto w = (1.0L/3.0L)*sqrt(3)*tw2.sqrt();
double dv3 = d1*d2*d3 - 1.0L/4.0L*pow(qe, 3) + (3.0L/4.0L)*pow(qe, 2)*(CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3) - 1.0L/4.0L*qe*(6*CosSq12*CosSq13*d2*d3 + 6*CosSq13*SinSq12*d1*d3 + 6*SinSq13*d1*d2 + pow(d1, 2) + pow(d2, 2) + pow(d3, 2));
double tcos = cos((1.0L/3.0L)*acos((3.0L/2.0L)*dv3/(tw2*w)));
double Em = -w*(tcos + sqrt(-3*pow(tcos, 2) + 3));
double Ep = 2*tcos*w;
double E0 = -Em - Ep;
double tmp0 = d1*d2 + d1*d3 + d2*d3 - 3.0L/4.0L*pow(qe, 2) + (3.0L/2.0L)*qe*(CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3);
double sumxi0 = 3*pow(Em, 2) + tmp0;
double sumxi1 = 3*pow(E0, 2) + tmp0;
double sumxi2 = 3*pow(Ep, 2) + tmp0;
double tmp1 = CosSq12*CosSq13*d1 + CosSq13*SinSq12*d2 + SinSq13*d3;
double tmp2 = CosSq12*CosSq13*d2*d3 + CosSq13*SinSq12*d1*d3 + SinSq13*d1*d2 + (1.0L/4.0L)*pow(qe, 2) + (1.0L/2.0L)*qe*(-CosSq12*CosSq13*d1 + CosSq12*CosSq13*d2 - CosSq13*d2 + CosSq13*d3 + d1 + d2);
double xi0_0 = (pow(Em, 2) - Em*qe + Em*tmp1 + tmp2)/sumxi0;
double xi0_1 = (pow(E0, 2) - E0*qe + E0*tmp1 + tmp2)/sumxi1;
double xi0_2 = (pow(Ep, 2) - Ep*qe + Ep*tmp1 + tmp2)/sumxi2;
double tmp3 = 2*Cos12*Cos23*Sin12*Sin13*Sin23*d1*cos(deltacp) - 2*Cos12*Cos23*Sin12*Sin13*Sin23*d2*cos(deltacp) + CosSq12*CosSq23*d2 + CosSq12*SinSq13*SinSq23*d1 + CosSq13*SinSq23*d3 + CosSq23*SinSq12*d1 + SinSq12*SinSq13*SinSq23*d2;
double tmp4 = CosSq13*SinSq23*d1*d2 + d1*d3*(-2*Cos12*Cos23*Sin12*Sin13*Sin23*cos(deltacp) + SinSq12*SinSq13*SinSq23 + SinSq12*SinSq23 - SinSq12 - SinSq23 + 1) + d2*d3*(2*Cos12*Cos23*Sin12*Sin13*Sin23*cos(deltacp) - SinSq12*SinSq13*SinSq23 - SinSq12*SinSq23 + SinSq12 + SinSq13*SinSq23) - 1.0L/2.0L*pow(qe, 2) - 1.0L/2.0L*qe*(-4*Cos12*Cos23*Sin12*Sin13*Sin23*d1*cos(deltacp) + 4*Cos12*Cos23*Sin12*Sin13*Sin23*d2*cos(deltacp) - CosSq12*CosSq13*d1 + 2*CosSq12*CosSq23*SinSq13*d1 + 2*CosSq12*SinSq23*d2 + 2*CosSq13*CosSq23*d3 - CosSq13*SinSq12*d2 + 2*CosSq23*SinSq12*SinSq13*d2 + 2*SinSq12*SinSq23*d1 - SinSq13*d3);
double xi1_0 = (pow(Em, 2) + (1.0L/2.0L)*Em*qe + Em*tmp3 + tmp4)/sumxi0;
double xi1_1 = (pow(E0, 2) + (1.0L/2.0L)*E0*qe + E0*tmp3 + tmp4)/sumxi1;
double xi1_2 = (pow(Ep, 2) + (1.0L/2.0L)*Ep*qe + Ep*tmp3 + tmp4)/sumxi2;
if (sflavor == fflavor) {
  if (sflavor == NeutrinoTraits::Muon) {
    double pconst = pow(xi1_0, 2) + pow(xi1_1, 2) + pow(xi1_2, 2);
    double p01 = 2*cos((1.0L/2.0L)*L*(E0 - Em)/E)*fabs(xi1_0*xi1_1);
    double p02 = 2*cos((1.0L/2.0L)*L*(Em - Ep)/E)*fabs(xi1_0*xi1_2);
    double p12 = 2*cos((1.0L/2.0L)*L*(E0 - Ep)/E)*fabs(xi1_1*xi1_2);
    double poscill = p01 + p02 + p12;
    res = pconst + poscill;
  }
  if (sflavor == NeutrinoTraits::Electron) {
    double pconst = pow(xi0_0, 2) + pow(xi0_1, 2) + pow(xi0_2, 2);
    double p01 = 2*cos((1.0L/2.0L)*L*(E0 - Em)/E)*fabs(xi0_0*xi0_1);
    double p02 = 2*cos((1.0L/2.0L)*L*(Em - Ep)/E)*fabs(xi0_0*xi0_2);
    double p12 = 2*cos((1.0L/2.0L)*L*(E0 - Ep)/E)*fabs(xi0_1*xi0_2);
    double poscill = p01 + p02 + p12;
    res = pconst + poscill;
  }
} else {
  double pconst = xi0_0*xi1_0 + xi0_1*xi1_1 + xi0_2*xi1_2;
  double tmp5 = Cos13*(-Cos12*Cos23*Sin12*d1 + Cos12*Cos23*Sin12*d2 - CosSq12*Sin13*Sin23*d1*cos(deltacp) - Sin13*Sin23*SinSq12*d2*cos(deltacp) + Sin13*Sin23*d3*cos(deltacp));
  double tmp6 = Cos13*(-Cos12*d2*d3*(Cos12*Sin13*Sin23*cos(deltacp) + Cos23*Sin12) + Sin12*d1*d3*(Cos12*Cos23 - Sin12*Sin13*Sin23*cos(deltacp)) + Sin13*Sin23*d1*d2*cos(deltacp));
  double tmp7 = Cos13*Sin13*Sin23*(CosSq12*d1 + SinSq12*d2 - d3)*sin(deltacp);
  double tmp8 = Cos13*Sin13*Sin23*(CosSq12*d2*d3 + SinSq12*d1*d3 - d1*d2)*sin(deltacp);
  double B0re = Em*tmp5 - 1.0L/2.0L*qe*tmp5 + tmp6;
  double B0im = Em*tmp7 - 1.0L/2.0L*qe*tmp7 + tmp8;
  double B0abs = sqrt(pow(B0im, 2) + pow(B0re, 2));
  double cosPsi0 = 0;
  double sinPsi0 = 0;
  if (fabs(B0abs) > 1E-10) {
    cosPsi0 = B0re/B0abs;
    sinPsi0 = B0im/B0abs;
  }
  double B1re = E0*tmp5 - 1.0L/2.0L*qe*tmp5 + tmp6;
  double B1im = E0*tmp7 - 1.0L/2.0L*qe*tmp7 + tmp8;
  double B1abs = sqrt(pow(B1im, 2) + pow(B1re, 2));
  double cosPsi1 = 0;
  double sinPsi1 = 0;
  if (fabs(B1abs) > 1E-10) {
    cosPsi1 = B1re/B1abs;
    sinPsi1 = B1im/B1abs;
  }
  double B2re = Ep*tmp5 - 1.0L/2.0L*qe*tmp5 + tmp6;
  double B2im = Ep*tmp7 - 1.0L/2.0L*qe*tmp7 + tmp8;
  double B2abs = sqrt(pow(B2im, 2) + pow(B2re, 2));
  double cosPsi2 = 0;
  double sinPsi2 = 0;
  if (fabs(B2abs) > 1E-10) {
    cosPsi2 = B2re/B2abs;
    sinPsi2 = B2im/B2abs;
  }
  double p01 = 0;
  if ((fabs(B0abs) > 1E-10) && (fabs(B1abs) > 1E-10)) {
    double phase01 = -1.0L/2.0L*L*(E0 - Em)/E;
    p01 = -2*sqrt(xi0_0*xi0_1*xi1_0*xi1_1)*(fsign*(cosPsi0*sinPsi1 - cosPsi1*sinPsi0)*sin(phase01) + (cosPsi0*cosPsi1 + sinPsi0*sinPsi1)*cos(phase01));
  }
  double p02 = 0;
  if ((fabs(B0abs) > 1E-10) && (fabs(B2abs) > 1E-10)) {
    double phase02 = (1.0L/2.0L)*L*(Em - Ep)/E;
    p02 = 2*sqrt(xi0_0*xi0_2*xi1_0*xi1_2)*(fsign*(cosPsi0*sinPsi2 - cosPsi2*sinPsi0)*sin(phase02) + (cosPsi0*cosPsi2 + sinPsi0*sinPsi2)*cos(phase02));
  }
  double p12 = 0;
  if ((fabs(B1abs) > 1E-10) && (fabs(B2abs) > 1E-10)) {
    double phase12 = (1.0L/2.0L)*L*(E0 - Ep)/E;
    p12 = -2*sqrt(xi0_1*xi0_2*xi1_1*xi1_2)*(fsign*(cosPsi1*sinPsi2 - cosPsi2*sinPsi1)*sin(phase12) + (cosPsi1*cosPsi2 + sinPsi1*sinPsi2)*cos(phase12));
  }
  double poscill = p01 + p02 + p12;
  res = pconst + poscill;
}

    // End of the COPY-PASTE

    return res;
  }


};
