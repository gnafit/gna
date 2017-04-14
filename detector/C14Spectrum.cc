#include "C14Spectrum.hh"

#include "FakeGSLFunction.hh"
#include "Units.hh"
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <boost/math/constants/constants.hpp>
#include <cassert>
#include <iostream>

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
constexpr auto MeV_to_keV = 1e3; 
constexpr auto keV_to_MeV = 1e-3; 
constexpr auto coincidence_window = 300*ns;
constexpr auto C14_half_life = 5730*year;
/* constexpr auto H_to_C_ratio = 1.639;    */



C14Spectrum::C14Spectrum(int order, int n_pivots): integration_order(order), n_pivots(n_pivots), coincidence_prob(0.), stayed_in_bin(0.)
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

    static const auto norm = 1./IntegrateSpectrum(spectrum_start_point, spectrum_end_point);   

/* Linear alkybenzene's chemical formula is C_6 H_5 C_n H_{2n+1}
 * where n is integer between 10 and 13 (see JUNO Conceptual Design Report), so convesrion between number of
 * protons (only hydrogen) to number of C12s is a bit unclear.
 * Here for no particular reasons we use the median n = 11*/

    decltype(auto) convers_H_to_C12 = [this](const double n){ return ((n+6)/(2*n+6) * this->m_protons); }; 
    double C14_nuclei_number = m_rho * convers_H_to_C12(11);
    coincidence_prob = (coincidence_window / C14_half_life) * C14_nuclei_number;

    int cachekey = 0;

    auto bin_width = m_datatype.edges[1] - m_datatype.edges[0];
    auto pivot_width = bin_width/n_pivots;
    
    if (this->stayed_in_bin == 0.)
    for (int pivot = 1; pivot <= n_pivots; ++pivot) {
        auto Etrue_piv = m_datatype.edges[0] + (2*pivot-1)*pivot_width/2;
        assert(Etrue_piv <= m_datatype.edges[1]); 
        /* std::cout << "Bin width " << m_datatype.edges[1] - m_datatype.edges[0] << std::endl;
         * std::cout << "Center of subbin " << (Etrue_piv - m_datatype.edges[0])*MeV_to_keV <<std::endl;
         * std::cout << "Integrating from 0 to "<< (m_datatype.edges[1] - Etrue_piv)*MeV_to_keV << std::endl;  */
        this->stayed_in_bin += norm  / n_pivots * IntegrateSpectrum(0, m_datatype.edges[1] - Etrue_piv); 
    } 

    for (size_t etrue = 0; etrue < m_size; ++etrue) {
        int startidx{-1};
        int endidx{-1};

        int bin_count{0};
        auto overall = 0.;
        double prob_acc{0.};
        for (size_t erec = etrue + 1; erec < m_size; ++erec) {


            /* for current pivot in bin "etrue" compute the relative probability for event to be shifted to bin "erec" */
            double rel_prob_sum{0.};
            double pivot_to_bin{0.};
            for (int pivot = 1; pivot <= n_pivots; ++pivot) {
                auto Etrue_piv = m_datatype.edges[etrue] + (2*pivot-1)*pivot_width/2;
                assert(Etrue_piv <= m_datatype.edges[etrue+1]);
                if ((m_datatype.edges[erec] - Etrue_piv)> spectrum_end_point) continue;

                auto current_start = m_datatype.edges[erec] - Etrue_piv;
                bool bin_fully_in_spectrum = (m_datatype.edges[erec+1] - Etrue_piv) < spectrum_end_point;
                auto current_end = bin_fully_in_spectrum ? (m_datatype.edges[erec+1] - Etrue_piv) : spectrum_end_point;
                /* std::cout << "Current start " << current_start*MeV_to_keV << " end " << current_end*MeV_to_keV << std::endl;  */

                if (current_start >= spectrum_end_point ) break;

               rel_prob_sum +=  IntegrateSpectrum(current_start, current_end);
               ++pivot_to_bin;
            }

            if (startidx < 0) startidx = etrue;
            if (rel_prob_sum == 0.) {
               endidx = startidx + bin_count;
               break;
            }
            /* std::cout << "Rel probability to transit to other bin " << rel_prob_sum; */
            /* std::cout << "Probability " << rel_prob_sum * coincidence_prob * norm/n_pivots << std::endl; */
            buf[erec] = rel_prob_sum * coincidence_prob * norm/(pivot_to_bin);
            overall +=  rel_prob_sum * coincidence_prob * norm/(pivot_to_bin);
            prob_acc += norm/(pivot_to_bin)*rel_prob_sum;
            ++bin_count;
  
        } 
        buf[etrue] = 1 - overall;

        if (endidx < 0) endidx = m_size;

        if (startidx >= 0) {
            std::copy(std::make_move_iterator(&buf[startidx]),
                      std::make_move_iterator(&buf[endidx]), 
                      &m_rescache[cachekey]); 
            m_cacheidx[etrue] = cachekey;
            cachekey += endidx - startidx;
        }
        else {
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
    assert(false);

    std::copy(events_true, events_true + insize, events_rec);
    double overall_moved = 0.;
    double total_number = 0.;

    double loss = 0.0;
    for (size_t etrue = 0; etrue < insize; ++etrue)
    {
        if (std::isnan(events_true[etrue])) continue;

        auto* cache = &m_rescache[m_cacheidx[etrue]];
        int startidx = m_startidx[etrue];
        int cnt = m_cacheidx[etrue + 1] - m_cacheidx[etrue];
        total_number += events_true[etrue];

        for (int offset = 0; offset < cnt; ++offset)
        {
            size_t erec = startidx + offset;
            if (erec == etrue) {
                continue;
            }
            auto rEvents = cache[offset];
            auto delta = rEvents * events_true[etrue];
            /* std::cout << "events_true["<<etrue<<"] " << events_true[etrue]
             * << " rEvents " << rEvents 
             * << " Delta " << delta << std::endl;   */

            events_rec[erec] += delta;
            overall_moved += delta;
            int inv = 2*etrue - erec;
            if (inv < 0 || (size_t)inv >= outsize)
            {
                events_rec[etrue] -= delta;
                loss += delta;
            }
        events_rec[etrue] -= delta;
        }
        
    } 
    /* std::cout << "overall_moved " << overall_moved << "\n" <<
     *              "how much we should move " << total_number * coincidence_prob * (1 -  stayed_in_bin) << std::endl; */


}

//--------------------------------------------------------------------------
inline double C14Spectrum::Spectra(double Ekin) const  noexcept
{
    assert(Ekin >= spectrum_start_point && Ekin <= spectrum_end_point);

    auto p =  std::sqrt(Ekin*Ekin + 2*Ekin*m_e);
    return Fermi_function(Ekin, p, Z_nitrogen, A_nitrogen) *
           std::pow((spectrum_end_point - Ekin), 2) * p * (Ekin + m_e); 
} 


double C14Spectrum::Fermi_function(double Ekin, double momentum, int Z, int A) const noexcept
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
    gsl_function_pp<decltype(ptr)> Fp(ptr);
    auto F = static_cast<gsl_function>(Fp);
    return gsl_integration_glfixed(&F, from, to, integr_table);
    
}


C14Spectrum::~C14Spectrum()
{
    gsl_integration_glfixed_table_free(integr_table);
}
