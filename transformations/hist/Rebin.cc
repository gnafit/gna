#include "Rebin.hh"
#include <algorithm>
#include <iterator>
#include <math.h>
#include "TypesFunctions.hh"

#include "config_vars.h"
#include "cuElementary.hh"
#include "GpuBasics.hh"

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
        std::cout << "im in! !!! !!! !!! "<<(this->m_new_edges.size()-1) * fargs.args[0].size() << std::endl;
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
  std::cout << "msparse: " << std::endl << m_sparse_cache <<std::endl;
  fargs.rets[0].x = m_sparse_cache * args[0].vec;
}

void Rebin::calcSmear_gpu(FunctionArgs& fargs) {
  fargs.args.touch();
  auto& args=fargs.args;
  auto& gpuargs = fargs.gpu;
  gpuargs->provideSignatureDevice();
  if( !m_initialized ){
      calcMatrix( args[0].type );
      Eigen::MatrixXd dMat = Eigen::MatrixXd(m_sparse_cache);
      double* tmp = dMat.data();
      copyH2D_NA(gpuargs->ints, tmp, fargs.ints[0].x.size());// (unsigned int)args[0].x.size() *(m_new_edges.size()-1));
  }
  std::cout << "m new - " << (this->m_new_edges.size()-1)  << ", ret size = " << fargs.rets[0].x.size() <<std::endl;
  curebin(gpuargs->args, gpuargs->ints,  gpuargs->rets, fargs.args[0].x.size(), fargs.rets[0].x.size());  
  //fargs.rets[0].x = m_sparse_cache * args[0].vec;
}


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
        throw std::runtime_error("Rebin: bin edges are not consistent (outer)");
      }
    }
    if(*edge_new!=*edge_old){
      throw std::runtime_error("Rebin: bin edges are not consistent (inner)");
    }
    ++edge_new;
  }
}

double Rebin::round(double num){
  return std::round( num*m_round_scale )/m_round_scale;
}
