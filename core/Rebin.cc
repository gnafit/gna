#include "Rebin.hh"

Rebin::Rebin(size_t n, double* edges, int rounding) : m_new_edges(n), m_round_scale(pow(10, rounding)) {
  m_new_edges.assign( edges, edges+n );

  transformation_(this, "smear")
      .input("histin")
      .output("histout")
      .types(&Rebin::calcMatrix)
      .func(&Rebin::calcSmear);
}

void Rebin::calcSmear(Args args, Rets rets) {
  rets[0].x = m_sparse_cache * args[0].vec;
}

void Rebin::calcMatrix(Atypes args, Rtypes rets) {
  if(!args[0].kind!=DataKind::Hist){
    throw std::runtime_error("Input should be a histogram");
  }
  rets[0]=DataType().hist().bins(m_new_edges.size()-1).edges(m_new_edges);

  m_sparse_cache.resize( rets[0].size(), args[0].size() );
  m_sparse_cache.setZero();
}

