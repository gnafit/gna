#include "C14_spectrum.hh"

#include "Units.hh"
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>
#include <boost/math/constants/constants.hpp>
#include "FakeGSLFunction.hh"

constexpr auto pi = boost::math::constants::pi<double>();
constexpr auto alpha = 1./137.035989;
/* constexpr auto fm_to_keV = 1 / (197.326 * 1e3);   */
constexpr auto fm_to_MeV = 1 / (197.326);  
constexpr auto Z_nitrogen = 7;
constexpr auto A_nitrogen = 14;
constexpr auto spectrum_end_point = 156.27 * 1e-3; /* MeV */
constexpr auto spectrum_start_point = 0.; /* MeV */
constexpr auto shape_factor = 1.24 ;   /* MeV^{-1} */
constexpr auto integration_order = 6;
/* constexpr auto MeV_to_keV = 1e3; */
constexpr auto coincidence_window = 300*ns;

C14_spectrum::C14_spectrum()
{
   variable_(&m_rho, "rho_C14");
   variable_(&m_protons, "TargetProtons");
   variable_(&m_e, "ElectronMass");
   callback_([this]{fillCache();});


   using namespace std::placeholders;
   transformation_(this, "smear")
                  .input("Nvis")
                  .output("NvisC")
                  .types(Atypes::pass<0>,
                         [](C14_spectrum *obj, Atypes args, Rtypes /*rets*/) {
                           obj->m_datatype = args[0];
                           obj->fillCache();
                         })
                  .func(&C14_spectrum::calcSmear);
   
}

/* Here we precalculate the matrix shifting the energy spectrum of IBD events, we take into account the coincidence of C14 events and IBD events in such a way */
void C14_spectrum::fillCache()
{
    m_size = m_datatype.hist().bins();
    if (m_size == 0) return;
    
    m_rescache.resize(m_size*m_size);
    m_cacheidx.resize(m_size + 1);
    m_startidx.resize(m_size);
    std::vector<double> buf(m_size);

    auto* table = gsl_integration_glfixed_table_alloc(integration_order);
    static const auto norm = 1/IntegrateSpectrum(0., spectrum_end_point, table); 
    
    
    for (size_t etrue = 0; etrue < m_size; ++etrue)
    {
        double Etrue = (m_datatype.edges[etrue+1] + m_datatype.edges[etrue])/2;
        double dEtrue = m_datatype.edges[etrue+1] - m_datatype.edges[etrue];

        int startidx = -1;
        int endidx = -1;

        double start_point = spectrum_start_point; 
        double end_point = Etrue + dEtrue;

        for (size_t erec = etrue; erec < m_size; ++erec)
        {
            double Erec = (m_datatype.edges[erec+1] + m_datatype.edges[erec])/2;
            double rEvents = dEtrue/norm  ;
            
        }

    } 



    gsl_integration_glfixed_table_free(table);
    


}


/* The implementation of C14 beta-spectra is to be here */
 double C14_spectrum::Spectra(double Ekin) const  noexcept
{
    auto p =  std::sqrt(Ekin*Ekin + 2*Ekin*m_e);
    return Fermi_function(Ekin, Z_nitrogen, A_nitrogen) *
           std::pow((spectrum_end_point - Ekin), 2) * p * (Ekin + m_e) ; 
} 


inline double C14_spectrum::Fermi_function(double Ekin, int Z, int A) const noexcept
{

    /* just using empirical formula for nuclear radius */
    auto r_nitrogen = 1.25 * std::cbrt(A) * fm_to_MeV;

    auto p =  std::sqrt(Ekin*Ekin + 2*Ekin*m_e);
    auto S =  std::sqrt(1 - std::pow(Z*alpha,2));
    auto eta =  alpha*(Ekin + m_e)*Z/p;

    gsl_sf_result lnGammaAbs, lnGammaArg, Gamma;
    gsl_sf_lngamma_complex_e( S, eta, &lnGammaAbs, &lnGammaArg);
    gsl_sf_gamma_e(1 + 2*S, &Gamma);

    return  2*(1+S) * std::exp(pi*eta)  * std::pow(2*p*r_nitrogen, 2*S - 2)
                    * std::pow(std::exp(lnGammaAbs.val)/Gamma.val, 2);
}

double C14_spectrum::IntegrateSpectrum(double from, double  to,
                                       GLTable* table)
{
    decltype(auto) ptr = [&](double x) {return this->Spectra(x);};
    gsl_function_wrapper<decltype(ptr)> Fp(ptr);
    auto F = static_cast<gsl_function>(Fp);
    return gsl_integration_glfixed(&F, from, to, table);
    
}


