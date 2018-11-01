#pragma once

#include <map>
#include <gsl/gsl_integration.h>

class GSLSamplerGL{
public:
   GSLSamplerGL(){};
  ~GSLSamplerGL();

  void fill(size_t n, double a, double b, double* x, double* w);
  void fill_bins(size_t nbins, int* orders, double* edges, double* abscissa, double* weight);

private:
  gsl_integration_glfixed_table* get_table(size_t n);
  std::map<size_t,gsl_integration_glfixed_table*> m_tables;
};

