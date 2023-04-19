#pragma once

#include <vector>

#include "GNAObject.hh"
#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

class EnergyResolutionInputs: public HistSmearSparse,
                              public TransformationBind<EnergyResolutionInputs> {
public:
    using TransformationBind<EnergyResolutionInputs>::transformation_;
    EnergyResolutionInputs(GNA::DataMutability input_edges_mode=GNA::DataMutability::Static);

    double resolution(double RelDelta, double Sigma) const noexcept;
private:
    void types(TypesFunctionArgs& fargs);
    void calcMatrix(FunctionArgs& fargs);
    void processEdgesIn();
    void processEdgesOut();
    std::vector<double> getOutputEdges() const noexcept final {
        return m_edges_out.size()>0u
            ? std::vector<double>(m_edges_out.data(), m_edges_out.data()+m_edges_out.size())
            : std::vector<double>{};
    }

    bool m_dynamic_edges{false};

    size_t m_nbins_in=0u;
    size_t m_nbins_out=0u;
    Eigen::ArrayXd m_edges_in;
    Eigen::ArrayXd m_edges_out;

    Eigen::ArrayXd m_widths_out;
    Eigen::ArrayXd m_centers_in;
    Eigen::ArrayXd m_centers_out;

    double m_cell_threshold = 1.e-9; /// The threshold below which the cell is assumed to be empty
    double m_delta_threshold;

    Eigen::ArrayXXd m_matrix;
};
