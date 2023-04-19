#pragma once

#include <vector>

#include "GNAObject.hh"
#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

class EnergyResolutionErfInputs: public HistSmearSparse,
                                 public TransformationBind<EnergyResolutionErfInputs> {
public:
    using TransformationBind<EnergyResolutionErfInputs>::transformation_;
    EnergyResolutionErfInputs(GNA::DataMutability input_edges_mode=GNA::DataMutability::Static);

private:
    void types(TypesFunctionArgs& fargs);
    void calcMatrix(FunctionArgs& fargs);
    void processEdges();
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

    double m_cell_threshold = 1.e-9; /// The threshold below which the cell is assumed to be empty

    Eigen::ArrayXXd m_Deltas;

    Eigen::ArrayXd  m_centers;
    Eigen::ArrayXd  m_widths;

    Eigen::ArrayXd  m_abssigmas_root2;

    Eigen::ArrayXXd m_deltas_rel_left;
    Eigen::ArrayXXd m_deltas_rel_right;

    Eigen::ArrayXXd m_part_erf_left;
    Eigen::ArrayXXd m_part_erf_right;
    Eigen::ArrayXXd m_part_exp_left;
    Eigen::ArrayXXd m_part_exp_right;

    Eigen::ArrayXXd m_matrix;
    Eigen::ArrayXd m_matrix_norm;
};
