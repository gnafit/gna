#include "HistNonlinearity.hh"
#include "TypesFunctions.hh"
#include <algorithm>
#include <fmt/format.h>

//#define DEBUG_ENL

#ifdef DEBUG_ENL
#define  DEBUG(...) do {                                      \
    printf(__VA_ARGS__);                                      \
  } while (0);
#else
#define  DEBUG(...)
#endif

HistNonlinearity::HistNonlinearity(bool single, bool propagate_matrix) :
HistSmearSparse(single, propagate_matrix)
{
  transformation_("matrix")
      .input("Edges")
      .input("EdgesModified")
      .output("FakeMatrix")
      .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>, TypesFunctions::ifSame)
      .types(TypesFunctions::edgesToMatrix<0,0,0>)
      .func(&HistNonlinearity::calcMatrix);

  if (single) {
    add(true);
  }
}

void HistNonlinearity::set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified ){
  if(!single()){
    throw std::runtime_error("HistNonlinearity::set(...) may be used only in 'single' mode");
  }
  if( m_initialized )
    throw std::runtime_error("HistNonlinearity is already initialized");
  m_initialized = true;

  transformations["matrix"].inputs[0].connect( bin_edges.single() );
  transformations["matrix"].inputs[1].connect( bin_edges_modified.single() );
}

void HistNonlinearity::set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified, SingleOutput& ntrue ){
  if(!single()){
    throw std::runtime_error("HistNonlinearity::set(...) may be used only in 'single' mode");
  }
  set( bin_edges, bin_edges_modified );
  transformations["smear"].inputs[0].connect( ntrue.single() );
}

void HistNonlinearity::set( SingleOutput& ntrue ){
  if(!single()){
    throw std::runtime_error("HistNonlinearity::set(...) may be used only in 'single' mode");
  }
  if( !m_initialized )
      throw std::runtime_error("HistNonlinearity is not initialized");

  transformations["smear"].inputs[0].connect( ntrue.single() );
}

void HistNonlinearity::set(){
  if(!single()){
    throw std::runtime_error("HistNonlinearity::set(...) may be used only in 'single' mode");
  }

  transformations["smear"].inputs[1].connect( transformations["matrix"].outputs[0] );
}

void HistNonlinearity::calcMatrix(FunctionArgs& fargs) {
  auto& args=fargs.args;
  auto n = args[0].arr.size();
  auto bins = n-1;
  auto* edges_orig = args[0].arr.data();
  auto* edges_mod  = args[1].arr.data();
  auto* end_orig = std::next(edges_orig, bins);
  auto* end_mod  = std::next(edges_mod, bins);

  m_sparse_cache.resize(bins, bins);
  m_sparse_cache.setZero();

  DEBUG("n=%li, matrix n=%i\n", n, int(n-1));

  // Find first bin in modified edge higher than lowest original value: set it as current bin
  auto* cur_bin = std::upper_bound( edges_mod, end_mod, m_range_min );
  DEBUG("found curbin: %li -> %g\n", std::distance(edges_mod, cur_bin), *cur_bin);
  if( *cur_bin<edges_orig[0] && cur_bin!=edges_mod ){
      cur_bin = std::prev(std::lower_bound( std::next(cur_bin), end_mod, edges_orig[0] ));
  }
  auto i_bin = cur_bin - edges_mod;
  if( cur_bin<end_mod ){
    // Find current bin's projection to the original range
    auto* cur_proj = std::lower_bound( edges_orig, end_orig, *cur_bin );
    if (cur_proj!=edges_orig) cur_proj = std::prev(cur_proj);
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
                 "%7li%6.2g""%7li%6.2g"
                 "%7li%s%6.2g""%7li%s%6.2g"
                 "%8.2g%1s%8.2g""%8.3f%8.3g %s\n",
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
    fargs.rets[0].mat = m_sparse_cache;
}

