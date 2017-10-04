#include "EnergyNonlinearity.hh"
#include <algorithm>

//FIXME:
#include <iostream>
using namespace std;

EnergyNonlinearity::EnergyNonlinearity() {
  transformation_(this, "smear")
      .input("Matrix")
      .input("Ntrue")
      .output("Nvis")
      .types(Atypes::pass<0>,
         [](EnergyNonlinearity *obj, Atypes args, Rtypes /*rets*/) {
           //obj->fillCache();
         })
       .func(&EnergyNonlinearity::calcSmear);

  transformation_(this, "matrix")
      .input("Edges")
      .input("EdgesModified")
      .output("Matrix")
      .types(Atypes::ifSame,
         [](EnergyNonlinearity *obj, Atypes args, Rtypes rets) {
           obj->m_size = args[0].shape[0]-1;
           obj->m_sparse_cache.resize(obj->m_size, obj->m_size);
           rets[0] = obj->m_datatype = DataType().points().shape( obj->m_size, obj->m_size );
         })
       .func(&EnergyNonlinearity::calcMatrix);
}

void EnergyNonlinearity::set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified, SingleOutput& ntrue ){
    if( m_initialized )
        throw std::runtime_error("EnergyNonlinearity is already initialized");
    m_initialized = true;

    t_["matrix"].inputs()[0].connect( bin_edges.single() );
    t_["matrix"].inputs()[1].connect( bin_edges_modified.single() );
    t_["smear"].inputs()[0].connect( t_["matrix"].outputs()[0] );
    t_["smear"].inputs()[1].connect( ntrue.single() );
}

void EnergyNonlinearity::calcSmear(Args args, Rets rets) {
  rets[0].x = m_sparse_cache * args[0].vec;
}

void EnergyNonlinearity::calcMatrix(Args args, Rets rets) {
  m_sparse_cache.setZero();

  auto n = args[0].arr.size();
  auto* edges_orig = args[0].arr.data();
  auto* edges_mod  = args[1].arr.data();
  auto* end_orig = std::next(edges_orig, n);
  auto* end_mod  = std::next(edges_mod, n);

  // Find first bin in modified edge higher than lowest original value: set it as current bin
  auto* cur_bin = std::upper_bound( edges_mod, end_mod, edges_orig[0] );
  auto i_bin = cur_bin - edges_mod;
  if( cur_bin<end_mod ){
    // Find current bin's projection to the original range
    auto* cur_proj = std::lower_bound( edges_orig, end_orig, *cur_bin );
    auto i_proj = cur_proj - edges_orig;
    if ( cur_proj<end_orig ){
      auto* next_bin = std::next(cur_bin);
      // Iterate bins
      while( cur_bin<end_mod ){
        auto full_width = *next_bin - *cur_bin;

        auto* cur_edge = cur_bin;
        // Iterate inner bin edges
        while ( cur_edge!=next_bin ){
          auto* next_edge = std::min( next_bin, std::next(cur_proj), [](auto a, auto b){ return *a<*b; } );

          m_sparse_cache.insert(i_bin, i_proj) = ( *next_edge - *cur_edge )/full_width;

          cur_edge = next_edge;
          std::advance(cur_proj, 1); i_proj++;
        }

        std::advance(cur_proj, -1); i_proj--;
        std::advance(cur_bin, 1); i_bin++;
        std::advance(next_bin, 1);
      }
    }
  }

  rets[0].mat = m_sparse_cache;
}

