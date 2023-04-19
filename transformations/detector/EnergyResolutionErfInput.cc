#include <boost/math/constants/constants.hpp>
#include "EnergyResolutionErfInput.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>
#include <string.h>

#include <unsupported/Eigen/SpecialFunctions>

// #define DEBUG_ERES_ERF
#ifdef DEBUG_ERES_ERF
    #include <iostream>
    using std::cout;
    using std::endl;
    #define DEBUG_START 0
    #define DEBUG_LEN 10
#endif

constexpr double half = boost::math::constants::half<double>();
constexpr double one_div_root_pi = boost::math::constants::one_div_root_pi<double>();
constexpr double root_two = boost::math::constants::root_two<double>();

using namespace TypeClasses;

EnergyResolutionErfInput::EnergyResolutionErfInput() :
HistSmearSparse(GNA::DataPropagation::Propagate)
{
    this->transformation_("matrix")
        .input("Edges", /*inactive*/true)     // Input bin edges [N]
        .input("RelSigma")                    // Relative Sigma value for each bin center [N-1]
        .output("FakeMatrix")
        .types(new CheckKindT<double>(DataKind::Hist, 0), new CheckKindT<double>(DataKind::Points, 1))
        .types(new CheckNdimT<double>(1), new CheckSameTypesT<double>({0,1}, "shape"))
        .types(&EnergyResolutionErfInput::types)
        .func(&EnergyResolutionErfInput::calcMatrix);

    add_transformation();
    add_input();
    set_open_input();
}

void EnergyResolutionErfInput::calcMatrix(FunctionArgs& fargs) {
    auto& args = fargs.args;

    auto cell_threshold = m_cell_threshold;
    auto clean = [cell_threshold](double value) noexcept -> double {
        return value>cell_threshold ? value : 0.0;
    };
    auto zero_to_one = [](double value) noexcept -> double {
        return value==0.0 ? 1.0 : value;
    };

    auto& relsigmas = args[1].arr;
    auto& abssigmas_root2 = root_two*(relsigmas*m_centers);
    auto nbins = relsigmas.size();

    auto nedges = nbins+1;
    auto m_Deltas_left  = m_Deltas.block(0, 0, nedges, nbins);
    auto m_Deltas_right = m_Deltas.block(0, 1, nedges, nbins);

    m_deltas_rel_left  = m_Deltas_left;
    m_deltas_rel_right = m_Deltas_right;

    m_deltas_rel_left.rowwise() /=abssigmas_root2.transpose();
    m_deltas_rel_right.rowwise()/=abssigmas_root2.transpose();

    m_part_erf_left  = m_Deltas_left  * Eigen::erf(m_deltas_rel_left);
    m_part_erf_right = m_Deltas_right * Eigen::erf(m_deltas_rel_right);
    m_part_exp_left  = (-m_deltas_rel_left.array().square()).exp();
    m_part_exp_right = (-m_deltas_rel_right.array().square()).exp();

    m_matrix = m_part_exp_left .block(1, 0, nbins, nbins)
             - m_part_exp_left .block(0, 0, nbins, nbins)
             - m_part_exp_right.block(1, 0, nbins, nbins)
             + m_part_exp_right.block(0, 0, nbins, nbins)
             ;
    m_matrix.rowwise()*=(half*one_div_root_pi)*abssigmas_root2.transpose();
    m_matrix+=half*(
                    m_part_erf_left .block(1, 0, nbins, nbins)
                  - m_part_erf_left .block(0, 0, nbins, nbins)
                  - m_part_erf_right.block(1, 0, nbins, nbins)
                  + m_part_erf_right.block(0, 0, nbins, nbins)
              );
    m_matrix.rowwise()/=m_widths.transpose();
    m_matrix_norm = m_matrix.colwise().sum().unaryExpr(zero_to_one);

#ifdef DEBUG_ERES_ERF
    cout<<"Sigmas rel: "<<relsigmas.rows()<<" "<<relsigmas.cols()<<"\n"<<relsigmas.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;
    cout<<"Sigmas abs*√2: "<<abssigmas_root2.rows()<<" "<<abssigmas_root2.cols()<<"\n"<<abssigmas_root2.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;

    cout<<"deltas rel (left): "<<m_deltas_rel_left.rows()<<" "<<m_deltas_rel_left.cols()<<"\n"<<m_deltas_rel_left.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN)<<endl;
    cout<<"deltas rel (right): "<<m_deltas_rel_right.rows()<<" "<<m_deltas_rel_right.cols()<<"\n"<<m_deltas_rel_right.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN)<<endl;

    cout<<"Erf (left): "<<m_part_erf_left.rows()<<" "<<m_part_erf_left.cols()<<"\n"<<m_part_erf_left.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN)<<endl;
    cout<<"Erf (right): "<<m_part_erf_right.rows()<<" "<<m_part_erf_right.cols()<<"\n"<<m_part_erf_right.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN)<<endl;
    cout<<"Exp (left): "<<m_part_exp_left.rows()<<" "<<m_part_exp_left.cols()<<"\n"<<m_part_exp_left.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN)<<endl;
    cout<<"Exp (right): "<<m_part_exp_right.rows()<<" "<<m_part_exp_right.cols()<<"\n"<<m_part_exp_right.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN)<<endl;

    cout<<"Matrix new (all): "<<m_matrix.rows()<<" "<<m_matrix.cols()<<"\n"<<m_matrix.block(DEBUG_START, DEBUG_START, DEBUG_LEN, DEBUG_LEN)<<endl;

    cout<<"1-matrix norm (all): "<<m_matrix_norm.rows()<<" "<<m_matrix_norm.cols()<<"\n"<<(1-m_matrix_norm).transpose()<<endl;
#endif

    m_matrix = m_matrix.unaryExpr(clean);
    for(int icol=0u; icol<nbins; icol++){
        if(m_matrix(0, icol)!=0.0 || m_matrix(nbins-1, icol)!=0.0){
            continue;
        }
        auto colsum=m_matrix_norm(icol);
        if (colsum>0.0){
            m_matrix.col(icol)/=colsum;
        }
    }

#ifdef DEBUG_ERES_ERF
    cout<<"Matrix new (clean): "<<m_matrix.rows()<<" "<<m_matrix.cols()<<"\n"<<m_matrix.block(DEBUG_START, DEBUG_START, DEBUG_LEN, DEBUG_LEN)<<endl;
    cout<<"1-Matrix renorm: "<<m_matrix_norm.rows()<<" "<<m_matrix_norm.cols()<<"\n"<<(1.0-m_matrix.colwise().sum()/m_matrix_norm.transpose())<<endl;
    cout<<"1-matrix norm (clean): "<<m_matrix_norm.rows()<<" "<<m_matrix_norm.cols()<<"\n"<<(1.0-m_matrix_norm).transpose()<<endl;
    cout<<endl;
#endif
    m_matrix_norm = m_matrix.colwise().sum();

    m_sparse_cache.resize(nbins, nbins);
    m_sparse_cache.setZero();
    m_sparse_cache = m_matrix.matrix().sparseView();
    m_sparse_cache.makeCompressed();
}

void EnergyResolutionErfInput::types(TypesFunctionArgs& fargs) {
    auto& etype = fargs.args[0];
    if(!etype.defined()){
        return;
    }

    auto nedges = etype.edges.size();
    auto nbins = nedges-1;

    auto& ret = fargs.rets[0];
    m_matrix.resize(nbins, nbins);
    ret.points().shape(nbins, nbins).preallocated(m_matrix.data());

    auto edges_a = Eigen::Map<const Eigen::ArrayXd>(etype.edges.data(), nedges);
    m_Deltas.resize(nedges, nedges);
    m_Deltas.colwise() = edges_a;
    m_Deltas.rowwise()-= edges_a.transpose();

    m_centers = half*(edges_a.tail(nbins) + edges_a.head(nbins));
    m_widths = edges_a.tail(nbins) - edges_a.head(nbins);

#ifdef DEBUG_ERES_ERF
    cout<<"Edges: "<<nedges<<"\n"<<edges_a.transpose().segment(DEBUG_START, DEBUG_LEN+1)<<endl;
    cout<<"Centers: "<<m_centers.rows()<<" "<<m_centers.cols()<<"\n"<<m_centers.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;
    cout<<"Widths: "<<m_widths.rows()<<" "<<m_widths.cols()<<"\n"<<m_widths.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;
    cout<<"Deltas (abs): "<<m_Deltas.rows()<<" "<<m_Deltas.cols()<<"\n"<<m_Deltas.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN+1)<<endl;
#endif
}
