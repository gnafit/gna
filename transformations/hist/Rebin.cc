#include "Rebin.hh"
#include <algorithm>
#include <iterator>
#include <math.h>
#include "TypesFunctions.hh"

#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#include "GpuBasics.hh"
#endif

Rebin::Rebin(size_t n, double* edges, int rounding) : m_new_edges(n), m_round_scale{pow(10, rounding)} {
  std::transform( edges, edges+n, m_new_edges.begin(), [this](double num){return this->round(num);} );

  transformation_("rebin")
    .input("histin")
    .output("histout")
    .types(TypesFunctions::ifHist<0>, [](Rebin *obj, TypesFunctionArgs& fargs){
           fargs.rets[0]=DataType().hist().edges(obj->m_new_edges);
           })
    .func(&Rebin::calcSmear)
#ifdef GNA_CUDA_SUPPORT
    .func("gpu", &Rebin::calcSmear_gpu, DataLocation::Device)
    .storage("gpu", [this](StorageTypesFunctionArgs& fargs) {
        fargs.ints[0] = DataType().points().shape((this->m_new_edges.size()-1) * fargs.args[0].size());
    })
#endif

    ;
}

void Rebin::calcSmear(FunctionArgs& fargs) {
  auto& args=fargs.args;
  if( !m_initialized ){
      calcMatrix( args[0].type );
  }
  fargs.rets[0].x = m_sparse_cache * args[0].vec;
}

#ifdef GNA_CUDA_SUPPORT
void Rebin::calcSmear_gpu(FunctionArgs& fargs) {
  fargs.args.touch();
  auto& args=fargs.args;
  auto& gpuargs = fargs.gpu;
  gpuargs->provideSignatureDevice();
  if( !m_initialized ){
      calcMatrix( args[0].type );
      fargs.ints[0].arr = Eigen::MatrixXd(m_sparse_cache);
      copyH2D_NA(gpuargs->ints, fargs.ints[0].arr.data(), fargs.ints[0].x.size());
  }
  curebin(gpuargs->args, gpuargs->ints,  gpuargs->rets, fargs.args[0].x.size(), fargs.rets[0].x.size());
}
#endif
void Rebin::calcMatrix(const DataType& type) {
  std::vector<double> edges(type.size()+1);
  std::transform( type.edges.begin(), type.edges.end(), edges.begin(), [this](double num){return this->round(num);} );

  m_sparse_cache.resize( m_new_edges.size()-1, type.size() );
  m_sparse_cache.setZero();

  auto edge_new = m_new_edges.begin();
  auto edge_old = std::lower_bound(edges.begin(), edges.end(), *edge_new);
  size_t iold=std::distance(edges.begin(), edge_old);
  for (size_t inew{0}; inew < m_new_edges.size(); ++inew) {
    while(*edge_old<*edge_new) {
      m_sparse_cache.insert(inew-1, iold) = 1.0;

      ++edge_old;
      ++iold;
      if(edge_old==edges.end()){
        dump(edges.size(), edges.data(), m_new_edges.size(), m_new_edges.data());
        throw std::runtime_error("Rebin: bin edges are not consistent (outer)");
      }
    }
    if(*edge_new!=*edge_old){
      dump(edges.size(), edges.data(), m_new_edges.size(), m_new_edges.data());
      throw std::runtime_error("Rebin: bin edges are not consistent (inner)");
    }
    ++edge_new;
  }
}

double Rebin::round(double num){
  return std::round( num*m_round_scale )/m_round_scale;
}

void Rebin::dump(size_t oldn, double* oldedges, size_t newn, double* newedges) const {
  std::cout<<"Rounding: "<<m_round_scale<<std::endl;
  std::cout<<"Old edges: " << Eigen::Map<const Eigen::Array<double,1,Eigen::Dynamic>>(oldedges, oldn)<<std::endl;
  std::cout<<"New edges: " << Eigen::Map<const Eigen::Array<double,1,Eigen::Dynamic>>(newedges, newn)<<std::endl;
}
