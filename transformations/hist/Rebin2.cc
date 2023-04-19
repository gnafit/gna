#include "Rebin2.hh"
#include <algorithm>
#include <iterator>
#include <math.h>
#include "TypesFunctions.hh"


Rebin2::Rebin2(size_t old_n, size_t new_n, double* old_edges, double* new_edges, int rounding) : m_old_edges(old_n), m_new_edges(new_n), m_round_scale{pow(10, rounding)} {
  std::transform( old_edges, old_edges+old_n, m_old_edges.begin(), [this](double num_old){return this->round(num_old);} );
  std::transform( new_edges, new_edges+new_n, m_new_edges.begin(), [this](double num_new){return this->round(num_new);} );

  transformation_("rebin")
    .input("array")
    .output("histout")
    //.types(TypesFunctions::ifHist<0>)
    .types(&Rebin2::calcMatrix)
    .func(&Rebin2::calcSmear);
}

void Rebin2::calcSmear(FunctionArgs& fargs) {
  auto& args=fargs.args;
  fargs.rets[0].x = m_sparse_cache * args[0].vec;
}


void Rebin2::calcMatrix(GNAObject::TypesFunctionArgs& fargs) {
  fargs.rets[0].hist().edges(m_new_edges);
  auto& type = fargs.args[0];
  if (type.size() != m_old_edges.size()-1){
      throw fargs.args.error(type, "Rebin2: bin edges are not consistent with input");
  }
  if(!type.defined()){
    return;
  }

  //std::vector<double> edges(type.size()+1);
  std::transform( type.edges.begin(), type.edges.end(), m_old_edges.begin(), [this](double num){return this->round(num);} );

  m_sparse_cache.resize( m_new_edges.size()-1, type.size() );
  m_sparse_cache.setZero();

  auto edge_new = m_new_edges.begin();
  auto edge_old = std::lower_bound(m_old_edges.begin(), m_old_edges.end(), *edge_new);
  size_t iold=std::distance(m_old_edges.begin(), edge_old);
  for (size_t inew{0}; inew < m_new_edges.size(); ++inew) {
    while(*edge_old<*edge_new) {
      m_sparse_cache.insert(inew-1, iold) = 1.0;

      ++edge_old;
      ++iold;
      if(edge_old==m_old_edges.end()){
        dump(m_old_edges.size(), m_old_edges.data(), m_new_edges.size(), m_new_edges.data());
        throw fargs.args.error(type, "Rebin2: bin edges are not consistent (outer)");
      }
    }
    if(*edge_new!=*edge_old){
      dump(m_old_edges.size(), m_old_edges.data(), m_new_edges.size(), m_new_edges.data());
      throw fargs.args.error(type, "Rebin2: bin edges are not consistent (inner)");
    }
    ++edge_new;
  }
}

double Rebin2::round(double num){
  return std::round( num*m_round_scale )/m_round_scale;
}

void Rebin2::dump(size_t oldn, double* oldedges, size_t newn, double* newedges) const {
  std::cout<<"Rounding: "<<m_round_scale<<std::endl;
  std::cout<<"Old edges: " << Eigen::Map<const Eigen::Array<double,1,Eigen::Dynamic>>(oldedges, oldn)<<std::endl;
  std::cout<<"New edges: " << Eigen::Map<const Eigen::Array<double,1,Eigen::Dynamic>>(newedges, newn)<<std::endl;
}
