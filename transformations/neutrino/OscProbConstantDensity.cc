#include "OscProbConstantDensity.hh"
#include <Eigen/Dense>
#include "OscillationVariables.hh"
#include "PMNSVariables.hh"
#include "TypesFunctions.hh"
#include "Units.hh"
#include <Eigen/src/Core/Matrix.h>
#include <complex>
#include <boost/math/constants/constants.hpp>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <fmt/ostream.h>
#include "Neutrino.hh"
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

struct element
{
    double per; // percentage
    int Z; // number of protons
    double A; // atomic mass
};

using namespace Eigen;
using namespace std::complex_literals;
using std::sin;
using std::cos;
using std::pow;
using std::sqrt;
using std::acos;
using std::exp;

template <typename T, std::size_t N>
using EigenValContainer = std::array<T, N>;

static constexpr double pi = boost::math::constants::pi<double>();

constexpr double compute_electron_num_earth () {

  // ELEMENTS IN EARTH'S CRUST
  // source of used Abundance of Elements in Earth’s Crust is
  // https://courses.lumenlearning.com/geology/chapter/reading-abundance-of-elements-in-earths-crust/

  constexpr auto num_elems = 10;
  constexpr std::array<element, num_elems> elems_in_crust =
    {{{0.466, 8, 15.999}, {0.277, 14, 28.086}, {0.081, 13, 26.982},
     {0.05, 26, 55.847}, {0.036, 20, 40.078}, {0.028, 11, 22.990},
     {0.026, 19, 39.098}, {0.021, 12, 24.305}, {0.004, 22, 47.867}, {0.011, 1, 1.008}} };

// effective mass of piece of crust in MeV
  constexpr double mass_crust_eff = [&](){
      double tmp = 0.;
      for(int j = 0; j < num_elems; ++j)
        tmp += elems_in_crust[j].per*elems_in_crust[j].A*NeutrinoUnits::aem;
      return tmp;}();

// effective number of electrons in mass_crust_eff
  constexpr double Ne_eff = [&](){
      double tmp = 0.;
      for(int j = 0; j < num_elems; ++j)
        tmp += elems_in_crust[j].per*elems_in_crust[j].Z;
      return tmp;}();

  constexpr auto density_to_MeV = NeutrinoUnits::g/(NeutrinoUnits::cm*NeutrinoUnits::cm*NeutrinoUnits::cm); // conversion coeff for density in g/cm^3 to MeV^4
  // ELECTRON NUMBER FROM CHEMICAL COMPOSITION
  constexpr auto Num_pieces = density_to_MeV/mass_crust_eff; // number of pieces of crust in rho_MeV in MeV^3 => [Num_pieces] = 1/volume
  constexpr auto Ne = Num_pieces*Ne_eff; // Ne in rho => [Ne] = 1/volume
  return Ne;
}

const double OscProbConstantDensity::rho_coeff = compute_electron_num_earth();

template <typename T>
EigenValContainer<T,5> EigenV(T p, T q) noexcept;

OscProbConstantDensity::OscProbConstantDensity(Neutrino from, Neutrino to, std::string str_distance, ExpAlgo algo): OscProbConstantDensity(from, to, str_distance)
{
    if (algo != ExpAlgo::putzer && algo != ExpAlgo::pade) {
        throw std::runtime_error("Only values of ExpAlgo enum are supported for choice of matrix exponential algorithms!");
    }
    m_exp_algo = algo;
}

OscProbConstantDensity::OscProbConstantDensity(Neutrino from, Neutrino to, std::string str_distance)
    : OscProbPMNSBase(from, to), m_from(from), m_to(to), m_str_distance(str_distance)
{
    if (from.kind != to.kind) {
      throw std::runtime_error("Particle-antiparticle oscillations");
    };

    variable_(&m_L, m_str_distance);
    variable_(&m_rho, "rho"); // g/cm3
    transformation_("oscprob")
        .input("Enu") //MeV
        .output("oscprob")
        .types(TypesFunctions::pass<0>)
        .func(&OscProbConstantDensity::calcOscProb);

    m_param->variable_("DeltaMSq12");
    m_param->variable_("DeltaMSq13");
    m_param->variable_("DeltaMSq23");
    m_param->variable_("Theta12");
    m_param->variable_("Theta13");
    m_param->variable_("Theta23");
    m_param->variable_("Alpha");
    m_param->variable_("Delta");

    m_alpha = from.flavor;
    m_beta = to.flavor;

    for (size_t i = 0; i < m_pmns->Nnu; ++i)
      for (size_t j = 0; j < m_pmns->Nnu; ++j)
        m_pmns->variable_(&m_pmns->V[i][j]);
}

void OscProbConstantDensity::calcOscProb(FunctionArgs fargs)
{
  /* fargs.args[0].x; --- array of input energies */
  /* fargs.rets[0].x; --- array of output probabilities, it should be filled in the algorithm */

  static constexpr auto coeff = sqrt(2.)*NeutrinoUnits::Gf;

  const auto Ne =  m_rho.value() * rho_coeff;
  const double rho = coeff*Ne; // rho in formulas with Magnus

  const double L = m_L*NeutrinoUnits::km; //km in MeV
  const auto& Enu = fargs.args[0].arr;
  auto& res = fargs.rets[0].arr;

  update_PMNS();

  for(int i = 0; i < Enu.size(); ++i)
    res[i] = __Matrix(Enu[i], rho, L);

}

void OscProbConstantDensity::update_PMNS() noexcept {
  for(size_t i = 0; i < m_pmns->Nnu; ++i)
    for(size_t j = 0; j < m_pmns->Nnu; ++j) {
      if (m_from.kind == Neutrino::Kind::Particle) {
          U(i,j) = m_pmns->V[i][j].complex();
      } else {
          U(i,j) = std::conj(m_pmns->V[i][j].complex());
      }
    }

  initial_state = {
    std::conj(U(m_alpha,0)),
    std::conj(U(m_alpha,1)),
    std::conj(U(m_alpha,2))
  };

  final_state = {
    std::conj(U(m_beta,0)),
    std::conj(U(m_beta,1)),
    std::conj(U(m_beta,2))
  };

}

double OscProbConstantDensity::__Matrix(double Enu, double rho, double L)
{

  Eigen::Matrix3cd H_z = Eigen::Matrix3cd::Zero();
  H_z(1,1) = (m_param->DeltaMSq12*NeutrinoUnits::eV*NeutrinoUnits::eV)/(2.*Enu);
  H_z(2,2) = (m_param->Alpha*m_param->DeltaMSq13*NeutrinoUnits::eV*NeutrinoUnits::eV)/(2.*Enu);

  Eigen::Matrix3cd V = Eigen::Matrix3cd::Zero();
  V(0,0) = 1.;

  const Eigen::Matrix3cd W = U.adjoint()*V*U;

  Eigen::Matrix3cd H = [&]()->Eigen::Matrix3cd {
      if (m_from.kind == Neutrino::Kind::Particle)
        return -(H_z + rho*W);
      else
        return -(H_z - rho*W);
  }();

  const Eigen::Matrix3cd S = [&]() -> Eigen::Matrix3cd {
      if (m_exp_algo == ExpAlgo::pade) {
          return (1i*L*H).exp();
      } else {
          const std::complex<double> trH = H(0,0) + H(1,1) + H(2,2); //trace of H

          const Eigen::Matrix3cd One = Eigen::Matrix3cd::Identity();

          const Eigen::Matrix3cd H_tr = H - (trH/3.)*One; // Make our hamiltonian traceless

          const Eigen::Matrix3cd Hsq = H_tr*H_tr; // Hsq = H_tr^2
          const auto detH = H_tr.determinant();
          const auto p = Hsq.trace()/2.; // trHsq*0.5

          const auto l = EigenV(p, detH);

          const auto F = 2.*sqrt(p/3.); // big parameter

          const auto r0 = - (2.*sin(l[3]*F*L/2.)*sin(l[3]*F*L/2.) - 1i*sin(l[3]*F*L))/l[3]; // r0/F
          const auto r1 = - (- r0 - (2.*sin(l[4]*F*L/2.)*sin(l[4]*F*L/2.) - 1i*sin(l[4]*F*L))/l[4])/(l[3] - l[4]); // r1/F^2
          return exp(1i*L*trH/3.)*exp(1i*l[0]*L*F)*( (1. - l[0]*(r0 - l[1]*r1) )*One + (r0 + l[2]*r1)*H_tr/F + r1*Hsq/(F*F));
      }
  }();

  const Vector3cd Psi = S*initial_state;

  const std::complex<double> dot_product = Psi.dot(final_state);
  const double Prob = std::norm(dot_product);

  return Prob;
}

template <typename T>
EigenValContainer<T, 5> EigenV(T p, T q) noexcept
{

  EigenValContainer<T, 5> l;

  T arg = acos((3.*q*sqrt(3.))/(2.*p*sqrt(p)))/3.;
  l[0] = cos(arg);
  l[1] = cos(arg - 2.*pi/3.);
  l[2] = cos(arg - 4.*pi/3.);

  // sort range [first, last), that is 3 is added to cbegin() iterator, the
  // last argument is function that is doing comparison between elements
  std::sort(l.begin(), l.begin()+3,
            [](const T& lhs, const T& rhs){return real(lhs) < real(rhs);});

  l[3] = l[1] - l[0]; //a
  l[4] = l[2] - l[0]; //b
  return l;
}
