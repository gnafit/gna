#ifndef QUADRATIC_H 
#define QUADRATIC_H

#include <cmath>

#include "GNAObject.hh"
#include "Eigen/Dense"
#include <algorithm>

class Quadratic: public GNAObject,
    public Transformation<Quadratic> {
        public:
            Quadratic() {
                variable_(&m_p0, "Qp0");
                variable_(&m_p1, "Qp1");
                variable_(&m_p2, "Qp2");

                transformation_(this, "QuaNL")
                    .input("old_bins")
                    .output("bins_after_nl")
                    .types(Atypes::pass<0,0>)
                    .func(&Quadratic::getNewBins)
                    ;
            }

            void getNewBins(Args args, Rets rets) {
                const auto& orig_bin_edges = args[0].arr;

                // Apply the conversion formula and get new bin edges. unaryExpr method applies given function to each element of Eigen array and modifies it inplace.
                // Here we took substituded each bin edges with its' square times weight.
                //new_bin_edges.unaryExpr( [&m_p0, m_p1, m_p2](auto& orig_bin_edges) {return m_p0+m_p1*orig_bin_edges+m_p2*orig_bin_edges*orig_bin_edges} );
                rets[0].arr = (m_p0+(1.0+m_p1)*orig_bin_edges+m_p2*orig_bin_edges*orig_bin_edges);

            }
        protected:
            variable<double> m_p0, m_p1, m_p2;
    };

#endif // QUADRATIC_H
