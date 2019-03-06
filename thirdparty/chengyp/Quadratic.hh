#pragma once

#include <cmath>

#include "GNAObject.hh"
#include "TypesFunctions.hh"
#include <Eigen/Dense>
#include <algorithm>

class Quadratic: public GNAObject,
                 public TransformationBind<Quadratic> {
        public:
            Quadratic() {
                variable_(&m_p0, "Qp0");
                variable_(&m_p1, "Qp1");
                variable_(&m_p2, "Qp2");

                this->transformation_("QuaNL")
                    .input("old_bins")
                    .output("bins_after_nl")
                    .types(TypesFunctions::pass<0,0>)
                    .func(&Quadratic::getNewBins)
                    ;
            }

            void getNewBins(FunctionArgs& fargs) {
                const auto& orig_bin_edges = fargs.args[0].arr;

                // Apply the conversion formula and get new bin edges. unaryExpr method applies given function to each element of Eigen array and modifies it inplace.
                // Here we took substituded each bin edges with its' square times weight.
                //new_bin_edges.unaryExpr( [&m_p0, m_p1, m_p2](auto& orig_bin_edges) {return m_p0+m_p1*orig_bin_edges+m_p2*orig_bin_edges*orig_bin_edges} );
                fargs.rets[0].arr = (m_p0+(1.0+m_p1)*orig_bin_edges+m_p2*orig_bin_edges*orig_bin_edges);

            }
        protected:
            variable<double> m_p0, m_p1, m_p2;
    };
