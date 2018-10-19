#include <gsl/gsl_integration.h>
#include <Eigen/Dense>
#include <map>
#include <iterator>

#include "SamplerGL.hh"
#include "TypesFunctions.hh"

using namespace Eigen;

class GSLSamplerGL{
public:
   GSLSamplerGL(){};
  ~GSLSamplerGL();

  void fill(size_t n, double a, double b, double* x, double* w);
private:
  gsl_integration_glfixed_table* get_table(size_t n);
  std::map<size_t,gsl_integration_glfixed_table*> m_tables;
};

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
void SamplerGL::init() {
  transformation_("points")
    .input("edges")
    .output("abscissas")
    .output("weights")
    .types(&TypesFunctions::if1d<1>, &SamplerGL::check)
    .func(&SamplerGL::compute)
    ;
}

void SamplerGL::check(TypesFunctionArgs& fargs){
  auto& rets=fargs.rets;
  size_t npoints=static_cast<size_t>(m_orders.sum());
  rets[0] = rets[1] = DataType().points().shape(npoints);

  if(fargs.args[0].shape[0]-1!=m_orders.size()){
    throw fargs.args.error(fargs.args[0], "Number of edges is inconsistent with number of bins");
  }
}

void SamplerGL::compute(FunctionArgs& fargs){
  auto& edges=fargs.args[0].x;
  auto& rets=fargs.rets;
  auto *abscissa(rets[0].buffer), *weight(rets[1].buffer);

  GSLSamplerGL sampler;
  for (size_t i = 0; i < m_orders.size(); ++i) {
    size_t n = m_orders[i];
    sampler.fill(n, edges[i], edges[i+1], abscissa, weight);
    std::advance(abscissa, n);
    std::advance(weight, n);
  }
}

