#ifndef EXPNONLINEARITY_H 
#define EXPNONLINEARITY_H

#include <cmath>

#include "GNAObject.hh"
#include "Eigen/Dense"
#include <algorithm>

class ExpNonlinearity: public GNAObject,
    public Transformation<ExpNonlinearity> {
        public:
            ExpNonlinearity() {
                variable_(&m_p0, "Exp_p0");
                variable_(&m_p1, "Exp_p1");

                transformation_(this, "ExpNL")
                    .input("old_bins")
                    .output("bins_after_nl")
                    .types(Atypes::pass<0,0>)
                    .func(&ExpNonlinearity::getNewBins)
                    ;
            }

            void getNewBins(Args args, Rets rets) {
                const auto& orig_bin_edges = args[0].arr;

                // Apply the conversion formula and get new bin edges. unaryExpr method applies given function to each element of Eigen array and modifies it inplace.
                // Here we took substituded each bin edges with its' square times weight.
                //new_bin_edges.unaryExpr( [&m_p0, m_p1, m_p2](auto& orig_bin_edges) {return m_p0+m_p1*orig_bin_edges+m_p2*orig_bin_edges*orig_bin_edges} );
                rets[0].arr = (1.0+m_p0)*((1+m_p1*exp(-0.2*orig_bin_edges)).inverse())*orig_bin_edges;

            }
        protected:
            variable<double> m_p0, m_p1;
    };

#endif // EXPNONLINEARITY_H
