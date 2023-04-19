#pragma once

#include <vector>

#include "GNAObject.hh"
#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

class EnergyResolutionErfInput: public HistSmearSparse,
                                public TransformationBind<EnergyResolutionErfInput> {
public:
    using TransformationBind<EnergyResolutionErfInput>::transformation_;
    EnergyResolutionErfInput();

private:
    void types(TypesFunctionArgs& fargs);
    void calcMatrix(FunctionArgs& fargs);

    double m_cell_threshold = 1.e-9; /// The threshold below which the cell is assumed to be empty

    Eigen::ArrayXXd m_Deltas;

    Eigen::ArrayXd  m_centers;
    Eigen::ArrayXd  m_widths;

    Eigen::ArrayXXd m_deltas_rel_left;
    Eigen::ArrayXXd m_deltas_rel_right;

    Eigen::ArrayXXd m_part_erf_left;
    Eigen::ArrayXXd m_part_erf_right;
    Eigen::ArrayXXd m_part_exp_left;
    Eigen::ArrayXXd m_part_exp_right;

    Eigen::ArrayXXd m_matrix;
    Eigen::ArrayXd m_matrix_norm;
};
