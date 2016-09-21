#include "C14Spectrum.hh"

#include "Units.hh"
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <boost/math/constants/constants.hpp>
#include "FakeGSLFunction.hh"
#include <cassert>
#include "ParametricLazy.hpp"

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
constexpr auto C14_half_life = 5730*year;
constexpr auto H_to_C_ratio = 1.639;

C14Spectrum::C14Spectrum(int order): integration_order(order)
{
   variable_(&m_rho, "rho_C14");
   variable_(&m_protons, "TargetProtons");
   variable_(&m_e, "ElectronMass");
   callback_([this]{fillCache();});


   integr_table = gsl_integration_glfixed_table_alloc(integration_order);


   using namespace std::placeholders;
   transformation_(this, "smear")
                  .input("Nvis")
                  .output("NvisC")
                  .types(Atypes::pass<0>,
                         [](C14Spectrum *obj, Atypes args, Rtypes /*rets*/) {
                           obj->m_datatype = args[0];
                           obj->fillCache();
                         })
                  .func(&C14Spectrum::calcSmear);
   
}

/* Here we precalculate the matrix shifting the energy spectrum of IBD events, we take into account the coincidence of C14 events and IBD events in such a way */
void C14Spectrum::fillCache()
{
    m_size = m_datatype.hist().bins();
    if (m_size == 0) return;
    
    m_rescache.resize(m_size*m_size);
    m_cacheidx.resize(m_size + 1);
    m_startidx.resize(m_size);
    std::vector<double> buf(m_size);

    static const auto norm = 1/IntegrateSpectrum(spectrum_start_point, spectrum_end_point);   

/* Linear alkybenzene's chemical formula is C_6 H_5 C_n H_{2n+1}
 * where n is integer between 10 and 13 (see JUNO Conceptual Design Report), so convesrion between number of
 * protons (only hydrogen) to number of C12s is a bit unclear.
 * Here for no particular reasons we use the median n = 11*/

    auto convers_H_to_C12 = [&](const int n){ return (n+6)/(2*n+6)*m_protons; }; 
    auto C14_nuclei_number = m_rho * convers_H_to_C12(11);
    auto coincidence_prob = coincidence_window * C14_half_life * C14_nuclei_number ;

    int cachekey = 0;
    
    for (size_t etrue = 0; etrue < m_size; ++etrue)
    {
        double Etrue = (m_datatype.edges[etrue+1] + m_datatype.edges[etrue])/2;

        int startidx = -1;
        int endidx = -1;

        /* check for how many bins do we spread events from a given one */
        auto current_start = 0.; 
        int bin_count = 0;

        auto overall = 0.;
        for (size_t erec = etrue + 1; erec < m_size; ++erec)
        {
            auto current_end = (m_datatype.edges[erec+1] + m_datatype.edges[erec])/2 - Etrue;
            bool bin_fully_in_spectrum = (current_end < spectrum_end_point);
            auto rEvents = bin_fully_in_spectrum ?
                           coincidence_prob * norm * IntegrateSpectrum(current_start, current_end) : 
                           coincidence_prob * norm * IntegrateSpectrum(current_start, spectrum_end_point);

            current_start = current_end;
            ++bin_count;

            buf[erec] = rEvents;
            overall +=  rEvents;

            if (startidx < 0) startidx = etrue;

            if (!bin_fully_in_spectrum)
            {
                endidx = startidx + bin_count;
                break;
            }
        }
        buf[etrue] = 1 - overall;

        if (endidx < 0) endidx = m_size;

        if (startidx >= 0)
        {
            std::copy(std::make_move_iterator(&buf[startidx]),
                      std::make_move_iterator(&buf[endidx]), 
                      &m_rescache[cachekey]); 
            m_cacheidx[etrue] = cachekey;
            cachekey += endidx - startidx;
        }
        else
        {
            m_cacheidx[etrue] = cachekey;
        }
        m_startidx[etrue] = startidx;
    }     
    m_cacheidx[m_size] = cachekey;
}


void C14Spectrum::calcSmear(Args args, Rets rets)
{
    const double* events_true = args[0].x.data();
    double* events_rec = rets[0].x.data();
    assert(events_rec != events_true);

    size_t insize = args[0].type.size();
    size_t outsize = rets[0].type.size();
    assert(insize == outsize);

    std::copy(events_true, events_true + insize, events_rec);

    double loss = 0.0;
    for (size_t etrue = 0; etrue < insize; ++etrue)
    {
        auto* cache = &m_rescache[m_cacheidx[etrue]];
        int startidx = m_startidx[etrue];
        int cnt = m_cacheidx[etrue + 1] - m_cacheidx[etrue];
        for (int off = 0; off < cnt; ++off)
        {
            size_t erec = startidx + off;
            if (erec == etrue)
            {
                continue;
            }
            auto rEvents = cache[off];
            auto delta = rEvents*events_true[etrue];

            events_rec[erec] += delta;
            int inv = 2*etrue - erec;
            if (inv < 0 || (size_t)inv >= outsize)
            {
                events_rec[etrue] -= delta;
                loss += delta;
            }
        events_rec[etrue] -= delta;
        }
        
    } 


}

//--------------------------------------------------------------------------
 double C14Spectrum::Spectra(double Ekin) const  noexcept
{
    assert(Ekin >= spectrum_start_point && Ekin <= spectrum_end_point);

    auto p =  std::sqrt(Ekin*Ekin + 2*Ekin*m_e);
    return Fermi_function(Ekin, p, Z_nitrogen, A_nitrogen) *
           std::pow((spectrum_end_point - Ekin), 2) * p * (Ekin + m_e); 
} 


inline double C14Spectrum::Fermi_function(double Ekin, double momentum, int Z, int A) const noexcept
{

    /* just using empirical formula for nuclear radius */
    auto r_nitrogen = 1.25 * std::cbrt(A) * fm_to_MeV;

    auto S =  std::sqrt(1 - std::pow(Z * alpha, 2));
    auto eta =  alpha * (Ekin + m_e) * Z / momentum;

    gsl_sf_result lnGammaAbs, lnGammaArg, Gamma;
    gsl_sf_lngamma_complex_e(S, eta, &lnGammaAbs, &lnGammaArg);
    gsl_sf_gamma_e(1 + 2*S, &Gamma);

    return  2*(1+S) * std::exp(pi*eta)  * std::pow(2*momentum*r_nitrogen, 2*S - 2)
                    * std::pow(std::exp(lnGammaAbs.val)/Gamma.val, 2);
}

double C14Spectrum::IntegrateSpectrum(double from, double  to)
{
    assert((to - from) <= spectrum_end_point);

    auto ptr = [this](double x) {return this->Spectra(x);};
    gsl_function_wrapper<decltype(ptr)> Fp(ptr);
    auto F = static_cast<gsl_function>(Fp);
    return gsl_integration_glfixed(&F, from, to, integr_table);
    
}


C14Spectrum::~C14Spectrum()
{
    gsl_integration_glfixed_table_free(integr_table);
}
