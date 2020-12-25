#include "GSLSamplerGL.hh"

#include <iterator>
using namespace std;

GSLSamplerGL::~GSLSamplerGL(){
  for(auto& kv: m_tables){
    gsl_integration_glfixed_table_free(kv.second);
  }
}

void GSLSamplerGL::fill(size_t n, double a, double b, double* x, double* w){
  if(!n){
    return;
  }
  auto* table=get_table(n);
  for (size_t j = 0; j<n; ++j) {
    gsl_integration_glfixed_point(a, b, j, x, w, table);
    std::advance(x,1);
    std::advance(w,1);
  }
}

void GSLSamplerGL::fill_bins(size_t nbins, int* order, double* edges, double* abscissa, double* weight){
  auto* edge_a=edges;
  auto* edge_b=next(edges);
  for (size_t i=0; i < nbins; ++i) {
    size_t n = static_cast<size_t>(*order);
    fill(n, *edge_a, *edge_b, abscissa, weight);
    advance(order, 1);
    advance(abscissa, n);
    advance(weight, n);
    advance(edge_a, 1);
    advance(edge_b, 1);
  }
}

gsl_integration_glfixed_table* GSLSamplerGL::get_table(size_t n){
    const auto& it=m_tables.find(n);
    if(it==m_tables.end()){
      return m_tables[n]=gsl_integration_glfixed_table_alloc(n);
    }
    return it->second;
}
