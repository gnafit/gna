#pragma once
#include <Eigen/Dense>
#include "GNAObject.hh"
#include "Neutrino.hh"
#include "OscProbPMNS.hh"

class OscProbConstantDensity: public OscProbPMNSBase,
                       public TransformationBind<OscProbConstantDensity> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using TransformationBind<OscProbConstantDensity>::transformation_;
    OscProbConstantDensity(Neutrino from, Neutrino to, std::string str_distance = "L");

    static const double rho_coeff;
    enum class ExpAlgo{putzer=1, pade=2};
    OscProbConstantDensity(Neutrino from, Neutrino to, std::string str_distance, ExpAlgo algo);

private:
    void calcOscProb(FunctionArgs fargs);
    double __Matrix(double Enu, double rho, double L);
    void update_PMNS() noexcept;

    Eigen::Matrix3cd U;
    Eigen::Vector3cd initial_state;
    Eigen::Vector3cd final_state;
    variable<double> m_L;
    variable<double> m_rho;
    Neutrino m_from;
    Neutrino m_to;
    std::string m_str_distance;
    ExpAlgo m_exp_algo = ExpAlgo::putzer;
};
