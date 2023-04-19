#pragma once

#include <vector>

#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

/**
 * @brief Rebin input histogram based on the binning, passed via input
 *
 * Operates via the sparse smearing matrix
 */
class RebinInput: public HistSmearSparse,
                  public TransformationBind<RebinInput> {
public:
    using TransformationBind<RebinInput>::transformation_;

    RebinInput(int rounding, GNA::DataMutability input_edges_mode=GNA::DataMutability::Static, GNA::DataPropagation propagate_matrix=GNA::DataPropagation::Ignore);

    void permitUnderflow() noexcept { m_permit_underflow=true; }
private:
    void calcMatrix(FunctionArgs& fargs);
    void getEdges(TypesFunctionArgs& fargs);
    void round(size_t n, const double* edges_raw, std::vector<double>& edges_rounded);
    void dump(size_t oldn, double* oldedges, size_t newn, double* newedges) const;
    std::vector<double> getOutputEdges() const noexcept final { return m_edges_out; }

    bool m_initialized{false};
    bool m_dynamic_edges{false};

    double m_round_scale;

    size_t m_nbins_in=0u;
    size_t m_nbins_out=0u;
    std::vector<double> m_edges_in;
    std::vector<double> m_edges_out;

    bool m_permit_underflow=false;
};
