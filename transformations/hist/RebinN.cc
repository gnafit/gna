#include "RebinN.hh"
#include "fmt/format.h"
#include <algorithm>
#include <iterator>
#include <math.h>

RebinN::RebinN(size_t n) : m_njoin(n) {
  transformation_("rebin")
    .input("histin")
    .output("histout")
    .types(TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>, &RebinN::doTypes)
    .func(&RebinN::doRebin);
}

void RebinN::doRebin(FunctionArgs& fargs) {
  auto& arg=fargs.args[0];
  auto& ret=fargs.rets[0];

  Eigen::Map<const Eigen::ArrayXXd> view(arg.buffer, ret.arr.size(), m_njoin);
  fargs.rets[0].x = view.rowwise().sum();
}

void RebinN::doTypes(TypesFunctionArgs& fargs) {
  auto& edges=fargs.args[0].edges;
  size_t nbins=edges.size()-1;
  if( nbins%m_njoin ){
    throw std::runtime_error(fmt::format("May not rebin {0} bins {1} times", nbins, m_njoin));
  }

  size_t nedges=nbins/m_njoin+1;
  std::vector<double> newedges(nedges);
  for (size_t i = 0; i < nedges; ++i) {
    newedges[i] = edges[i*m_njoin];
  }
  newedges[nedges-1]=edges.back();
  fargs.rets[0]=DataType().hist().edges(newedges);
}
