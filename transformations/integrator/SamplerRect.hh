#pragma once

#include <string>

class SamplerRect{
public:
  static int offset(std::string mode);
  static void fill(int offset, size_t order, double a, double b, double* x, double* w);
  static void fill_bins(int offset, size_t nbins, int* orders, double* edges, double* abscissa, double* weight);

private:
};

