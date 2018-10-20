#include "GSLSamplerGL.hh"

GSLSamplerGL::~GSLSamplerGL(){
  for(auto& kv: m_tables){
    gsl_integration_glfixed_table_free(kv.second);
  }
}

void GSLSamplerGL::fill(size_t n, double a, double b, double* x, double* w){
  auto* table=get_table(n);
  for (size_t j = 0; j<n; ++j) {
    gsl_integration_glfixed_point(a, b, j, x, w, table);
    std::advance(x,1);
    std::advance(w,1);
  }
}

gsl_integration_glfixed_table* GSLSamplerGL::get_table(size_t n){
    const auto& it=m_tables.find(n);
    if(it==m_tables.end()){
      return m_tables[n]=gsl_integration_glfixed_table_alloc(n);
    }
    return it->second;
}
