#include "EnergyNonlinearity.hh"
#include <algorithm>

//#define DEBUG_ENL

#ifdef DEBUG_ENL
#define  DEBUG(...) do {                                      \
    printf(__VA_ARGS__);                                      \
  } while (0);
#else
#define  DEBUG(...)
#endif

EnergyNonlinearity::EnergyNonlinearity( bool propagate_matrix ) : m_propagate_matrix(propagate_matrix) {
  transformation_(this, "smear")
      .input("FakeMatrix")
      .input("Ntrue")
      .output("Nvis")
      .types(Atypes::pass<1,0>)
      .func(&EnergyNonlinearity::calcSmear);

  transformation_(this, "matrix")
      .input("Edges")
      .input("EdgesModified")
      .output("FakeMatrix")
      .types(Atypes::ifSame,
         [](EnergyNonlinearity *obj, Atypes args, Rtypes rets) {
         obj->m_size = args[0].shape[0]-1;
         obj->m_sparse_cache.resize(obj->m_size, obj->m_size);
         if( obj->m_propagate_matrix ){
           rets[0] = obj->m_datatype = DataType().points().shape( obj->m_size, obj->m_size );
         }
         else{
           rets[0] = obj->m_datatype = DataType().points().shape( 0, 0 );
         }
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
  args[0]; // Needed to trigger updating
  rets[0].x = m_sparse_cache * args[1].vec;
}

void EnergyNonlinearity::calcMatrix(Args args, Rets rets) {
  m_sparse_cache.setZero();

  auto n = args[0].arr.size();
  auto* edges_orig = args[0].arr.data();
  auto* edges_mod  = args[1].arr.data();
  auto* end_orig = std::next(edges_orig, n-1);
  auto* end_mod  = std::next(edges_mod, n-1);

  DEBUG("n=%li, matrix n=%li\n", n, m_size);

  // Find first bin in modified edge higher than lowest original value: set it as current bin
  auto* cur_bin = std::upper_bound( edges_mod, end_mod, m_range_min );
  if( *cur_bin<edges_orig[0] ){
      cur_bin = std::prev(std::lower_bound( std::next(cur_bin), end_mod, edges_orig[0] ));
  }
  auto i_bin = cur_bin - edges_mod;
  if( cur_bin<end_mod ){
    // Find current bin's projection to the original range
    auto* cur_proj = std::prev(std::lower_bound( edges_orig, end_orig, *cur_bin ));
    auto i_proj = cur_proj - edges_orig;
    if ( cur_proj<end_orig && cur_bin<end_mod ){
      auto* next_bin = std::next(cur_bin);
      // Iterate bins
      #ifdef DEBUG_ENL
      size_t iteration=0;
      #endif
      while( cur_bin<end_mod && *cur_bin<*end_orig && *next_bin<m_range_max ){
        auto full_width = *next_bin - *cur_bin;

        auto* cur_edge{cur_bin};
        #ifdef DEBUG_ENL
        bool cur_mod{true};
        #endif
        // Iterate inner bin edges
        while ( (cur_edge!=next_bin) && (cur_proj!=end_orig) ){
          auto* next_edge = std::min( next_bin, std::next(cur_proj), [](auto a, auto b){ return *a<*b; } );

          #ifdef DEBUG_ENL
          bool next_mod = next_edge==next_bin;
          #endif

          double f = ( *next_edge - *cur_edge )/full_width;

          #ifdef DEBUG_ENL
          if ( ((iteration++)%20)==0 ) {
          printf("\n%6s%13s%13s%14s%14s%8s %8s%8s%8s\n",
                "it",
                "curbin", "curproj", "curedge",
                "nextedge", "nextbin", "nextproj",
                "weight", "width");
          }
          printf("%6li"
                 "%7li%6.2f""%7li%6.2f"
                 "%7li%s%6.2f""%7li%s%6.2f"
                 "%8.2f%1s%8.2f""%8.3f%8.3f %s\n",
                 iteration,
                 i_bin, *cur_bin, i_proj, *cur_proj,
                 std::distance(cur_mod?edges_mod:edges_orig, cur_edge), cur_mod?"j":"i", *cur_edge,
                 std::distance(next_mod?edges_mod:edges_orig, next_edge), next_mod?"j":"i", *next_edge,
                 *next_bin, *next_bin==*std::next(cur_proj) ? "=":" ", *std::next(cur_proj),
                 f, full_width, f==0.0 ? "*" : "" );
          #endif
          m_sparse_cache.insert(i_proj, i_bin) = f;

          cur_edge = next_edge;
          std::advance(cur_proj, 1); i_proj++;
          #ifdef DEBUG_ENL
          cur_mod = next_mod;
          #endif
        }
        DEBUG("\n");

        if(*next_bin!=*cur_proj) {
          std::advance(cur_proj, -1); i_proj--;
        }
        std::advance(cur_bin, 1); i_bin++;
        std::advance(next_bin, 1);
      }
    }
  }
  DEBUG("\n");

  if ( m_propagate_matrix )
    rets[0].mat = m_sparse_cache;
}

