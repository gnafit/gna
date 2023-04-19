#include <boost/math/constants/constants.hpp>
#include "EnergyResolutionErfInputs.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>
#include <string.h>

#include <unsupported/Eigen/SpecialFunctions>

// #define DEBUG_ERES_ERFS
#ifdef DEBUG_ERES_ERFS
    #include <iostream>
    using std::cout;
    using std::endl;
    // #define DEBUG_START 230
    #define DEBUG_START 0
    #define DEBUG_LEN 10
#endif

constexpr double half = boost::math::constants::half<double>();
constexpr double one_div_root_pi = boost::math::constants::one_div_root_pi<double>();
constexpr double root_two = boost::math::constants::root_two<double>();

using GNA::DataMutability;
using GNA::DataPropagation;
using GNA::MatrixType;

EnergyResolutionErfInputs::EnergyResolutionErfInputs(DataMutability input_edges_mode) :
HistSmearSparse(DataPropagation::Propagate, MatrixType::Any),
m_dynamic_edges(input_edges_mode==DataMutability::Dynamic)
{
    using namespace TypeClasses;

    this->transformation_("matrix")
        .input("HistEdgesOut", /*inactive*/true)                                // Output bin edges [N]
        .input("EdgesIn", /*inactive*/input_edges_mode==DataMutability::Static) // Input bin edges [M]
        .input("RelSigma")                                                      // Relative Sigma value for each bin center [M-1]
        .output("FakeMatrix")
        .types(new CheckKindT<double>(DataKind::Hist, 0), new CheckKindT<double>(DataKind::Points, 2))
        .types(new CheckNdimT<double>(1))
        .types(&EnergyResolutionErfInputs::types)
        .func(&EnergyResolutionErfInputs::calcMatrix);

    add_transformation();
    add_input();
    set_open_input();
}

void EnergyResolutionErfInputs::calcMatrix(FunctionArgs& fargs) {
    auto& args = fargs.args;

    if(m_dynamic_edges && m_edges_in.size()==0u){
        m_edges_in = args[1].x;
        m_nbins_in = m_edges_in.size() - 1;
        processEdges();
    }

    auto cell_threshold = m_cell_threshold;
    auto clean = [cell_threshold](double value) noexcept -> double {
        return value>cell_threshold ? value : 0.0;
    };
    auto zero_to_one = [](double value) noexcept -> double {
        return value==0.0 ? 1.0 : value;
    };

    auto& relsigmas = args[2].arr;
    if(static_cast<size_t>(relsigmas.size())==m_nbins_in){
        // σ/E are computed at the bin centers: use as is
        m_abssigmas_root2 = root_two*(relsigmas*m_centers);
    }
    else{
        // σ/E are computed at the bin edges: use averages
        m_abssigmas_root2 = root_two*(relsigmas*m_centers);
        m_abssigmas_root2 = (0.5*root_two)*((relsigmas.head(m_nbins_in)+relsigmas.tail(m_nbins_in))*m_centers);
    }

    auto nedges_out = m_nbins_out+1;
    auto m_Deltas_left  = m_Deltas.block(0, 0, nedges_out, m_nbins_in);
    auto m_Deltas_right = m_Deltas.block(0, 1, nedges_out, m_nbins_in);

    m_deltas_rel_left  = m_Deltas_left;
    m_deltas_rel_right = m_Deltas_right;

    m_deltas_rel_left.rowwise() /=m_abssigmas_root2.transpose();
    m_deltas_rel_right.rowwise()/=m_abssigmas_root2.transpose();

    m_part_erf_left  = m_Deltas_left  * Eigen::erf(m_deltas_rel_left);
    m_part_erf_right = m_Deltas_right * Eigen::erf(m_deltas_rel_right);
    m_part_exp_left  = (-m_deltas_rel_left.array().square()).exp();
    m_part_exp_right = (-m_deltas_rel_right.array().square()).exp();

    m_matrix = m_part_exp_left .block(1, 0, m_nbins_out, m_nbins_in)
             - m_part_exp_left .block(0, 0, m_nbins_out, m_nbins_in)
             - m_part_exp_right.block(1, 0, m_nbins_out, m_nbins_in)
             + m_part_exp_right.block(0, 0, m_nbins_out, m_nbins_in)
             ;
    m_matrix.rowwise()*=(half*one_div_root_pi)*m_abssigmas_root2.transpose();
    m_matrix+=half*(
                    m_part_erf_left .block(1, 0, m_nbins_out, m_nbins_in)
                  - m_part_erf_left .block(0, 0, m_nbins_out, m_nbins_in)
                  - m_part_erf_right.block(1, 0, m_nbins_out, m_nbins_in)
                  + m_part_erf_right.block(0, 0, m_nbins_out, m_nbins_in)
              );
    m_matrix.rowwise()/=m_widths.transpose();
    m_matrix_norm = m_matrix.colwise().sum().unaryExpr(zero_to_one);

#ifdef DEBUG_ERES_ERFS
    cout<<"Sigmas rel: "<<relsigmas.rows()<<" "<<relsigmas.cols()<<"\n"<<relsigmas.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;
    cout<<"Sigmas abs*√2: "<<m_abssigmas_root2.rows()<<" "<<m_abssigmas_root2.cols()<<"\n"<<m_abssigmas_root2.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;

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
    for(size_t icol=0u; icol<m_nbins_in; icol++){
        if(m_matrix(0, icol)!=0.0 || m_matrix(m_nbins_out-1, icol)!=0.0){
            continue;
        }
        auto colsum=m_matrix_norm(icol);
        if (colsum>0.0){
            m_matrix.col(icol)/=colsum;
        }
    }

#ifdef DEBUG_ERES_ERFS
    cout<<"Matrix new (clean): "<<m_matrix.rows()<<" "<<m_matrix.cols()<<"\n"<<m_matrix.block(DEBUG_START, DEBUG_START, DEBUG_LEN, DEBUG_LEN)<<endl;
    cout<<"1-Matrix renorm: "<<m_matrix_norm.rows()<<" "<<m_matrix_norm.cols()<<"\n"<<(1.0-m_matrix.colwise().sum()/m_matrix_norm.transpose())<<endl;
    cout<<"1-matrix norm (clean): "<<m_matrix_norm.rows()<<" "<<m_matrix_norm.cols()<<"\n"<<(1.0-m_matrix_norm).transpose()<<endl;
    cout<<endl;
#endif
    m_matrix_norm = m_matrix.colwise().sum();

    m_sparse_cache.resize(m_nbins_out, m_nbins_in);
    m_sparse_cache.setZero();
    m_sparse_cache = m_matrix.matrix().sparseView();
    m_sparse_cache.makeCompressed();
}

void EnergyResolutionErfInputs::types(TypesFunctionArgs& fargs) {
    auto& args = fargs.args;

    auto& dtype_out = args[0];
    if(dtype_out.defined() && m_edges_out.size()==0u){
        m_nbins_out = dtype_out.size();
        m_edges_out = Eigen::Map<const Eigen::ArrayXd>(dtype_out.edges.data(), dtype_out.edges.size());
    }

    auto& dtype_in = args[1];
    switch(dtype_in.kind){
        case DataKind::Hist:
            if(m_edges_in.size()==0u) {
                m_nbins_in = dtype_in.size();
                m_edges_in = Eigen::Map<const Eigen::ArrayXd>(dtype_in.edges.data(), dtype_in.edges.size());

                if(m_dynamic_edges){
                    throw args.error(dtype_in, fmt::format("Arg 1 (EdgesIn) should not be a Histogram. Dynamic input is expected."));
                }
            }
            break;

        case DataKind::Points:
            m_nbins_in = dtype_in.size();
            if(m_nbins_in>1){
                m_nbins_in--;
            }
            else{
                throw args.error(dtype_in, fmt::format("Arg 1 (EdgesIn, points) should have at least 2 elements."));
            }
            if(!m_dynamic_edges){
                throw args.error(dtype_in, fmt::format("Arg 1 (EdgesIn) should not be a Points instance. Not dynamic input is expected."));
            }
            break;

        case DataKind::Undefined:
            break;
    }

    if(m_nbins_in>0u && m_nbins_out>0u){
        auto& ret = fargs.rets[0];
        m_matrix.resize(m_nbins_out, m_nbins_in);
        ret.points().shape(m_nbins_out, m_nbins_in).preallocated(m_matrix.data());

        if(m_edges_in.size()>0u){
            processEdges();
        }
    }

    auto& dtype_relsigma = args[2];
    if(m_nbins_in>0u && dtype_relsigma.defined()){
        auto n_rs = dtype_relsigma.size();
        if (n_rs!=m_nbins_in && n_rs!=(m_nbins_in+1)){
            throw args.error(dtype_relsigma, fmt::format("Arg 2 (RelSigma) should be of size of N bins (in) or N edges (in)."));
        }
    }
}

void EnergyResolutionErfInputs::processEdges(){
    m_Deltas.resize(m_edges_out.size(), m_edges_in.size());
    m_Deltas.colwise() = m_edges_out;
    m_Deltas.rowwise()-= m_edges_in.transpose();

    m_centers = half*(m_edges_in.tail(m_nbins_in) + m_edges_in.head(m_nbins_in));
    m_widths = m_edges_in.tail(m_nbins_in) - m_edges_in.head(m_nbins_in);

#ifdef DEBUG_ERES_ERFS
    cout<<"Edges in: "<<m_nbins_in+1u<<"\n"<<m_edges_in.transpose().segment(DEBUG_START, DEBUG_LEN+1)<<endl;
    cout<<"Edges out: "<<m_nbins_out+1u<<"\n"<<m_edges_out.transpose().segment(DEBUG_START, DEBUG_LEN+1)<<endl;
    cout<<"Centers (in): "<<m_centers.rows()<<" "<<m_centers.cols()<<"\n"<<m_centers.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;
    cout<<"Widths (in): "<<m_widths.rows()<<" "<<m_widths.cols()<<"\n"<<m_widths.transpose().segment(DEBUG_START, DEBUG_LEN)<<endl;
    cout<<"Deltas (out-in, abs): "<<m_Deltas.rows()<<" "<<m_Deltas.cols()<<"\n"<<m_Deltas.block(DEBUG_START, DEBUG_START, DEBUG_LEN+1, DEBUG_LEN+1)<<endl;
#endif
}

