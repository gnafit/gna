#pragma once

#include <vector>
#include <gsl/gsl_integration.h>

#include "GNAObject.hh"
#include "PDGVariables.hh"

using GLTable = gsl_integration_glfixed_table;

class C14Spectrum: public GNAObject,
                    public TransformationBind<C14Spectrum>
{
    public:
        C14Spectrum(int order, int n_pivots);
    private:
        ~C14Spectrum();
        inline double Fermi_function(double Ekin, double momentum, int Z, int A) const noexcept;
        double Spectra(double Ekin) const noexcept;
        void   fillCache();
        double IntegrateSpectrum(double from, double to);

        void   calcSmear(FunctionArgs fargs);


        variable<double> m_rho, m_protons, m_e, m_coinc_window;
        /* dependant<double> m_e; */

        DataType m_datatype;

        GLTable* integr_table;
        int integration_order;
        int n_pivots;

        double coincidence_prob;
        double stayed_in_bin;

        size_t m_size;
        std::vector<double> m_rescache;
        std::vector<int>    m_cacheidx;
        std::vector<int>    m_startidx;


};
