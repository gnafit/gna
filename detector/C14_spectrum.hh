#ifndef C14_spectrum_H
#define C14_spectrum_H

#include <vector>
#include <gsl/gsl_integration.h>

#include "GNAObject.hh"

using GLTable = gsl_integration_glfixed_table;

class C14_spectrum: public GNAObject,
                    public Transformation<C14_spectrum>
{
    public:
        C14_spectrum();
    private:
        inline double Fermi_function(double Ekin, int Z, int A) const noexcept;
        double Spectra(double Ekin) const noexcept;
        void   fillCache();
        double IntegrateSpectrum(double from, double to,
                                 GLTable* table);

        void   calcSmear(Args args, Rets rets);


        variable<double>    m_rho, m_protons, m_e;
        DataType m_datatype;

        size_t m_size;
        std::vector<double> m_rescache;
        std::vector<int>    m_cacheidx;
        std::vector<int>    m_startidx;
        

};

#endif
