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

HistNonlinearity::HistNonlinearity(bool propagate_matrix) :
HistSmearSparse(propagate_matrix)
{
  transformation_("matrix")
      .input("Edges", /*inactive*/true)
      .input("EdgesModified")
      .output("FakeMatrix")
      .types(TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>, TypesFunctions::ifBinsEdges<0,1>)
      .types(TypesFunctions::toMatrix<0,0,0>)
      .types(&HistNonlinearity::getEdges)
      .func(&HistNonlinearity::calcMatrix);

  add_transformation();
  add_input();
  set_open_input();
}

void HistNonlinearity::set(SingleOutput& bin_edges, SingleOutput& bin_edges_modified){
  if( m_initialized )
    throw std::runtime_error("HistNonlinearity is already initialized");
  m_initialized = true;

  auto inputs = transformations.front().inputs;
  bin_edges.single()          >> inputs[0];
  bin_edges_modified.single() >> inputs[1];
}

void HistNonlinearity::getEdges(TypesFunctionArgs& fargs) {
  m_edges = fargs.args[0].edges.data();
}

void HistNonlinearity::calcMatrix(FunctionArgs& fargs) {
  auto& args=fargs.args;
  size_t bins = args[1].arr.size()-1;
  auto* edges_orig = m_edges;
  auto* edges_mod  = args[1].arr.data();
  auto* end_orig = std::next(edges_orig, bins);
  auto* end_mod  = std::next(edges_mod, bins);

  m_sparse_cache.resize(bins, bins);
  m_sparse_cache.setZero();

  DEBUG("n=%zu, matrix n=%zu\n", bins+1, bins);

  // Find first bin in modified edge higher than lowest original value: set it as current bin
  auto* cur_bin_mod = std::upper_bound( edges_mod, end_mod, m_range_min );
  DEBUG("found curbin mod %li: %g -> %g\n", std::distance(edges_mod, cur_bin_mod), *cur_bin_mod, *(cur_bin_mod+1));
  if( *cur_bin_mod<edges_orig[0] && cur_bin_mod!=edges_mod ){
    cur_bin_mod = std::prev(std::lower_bound( std::next(cur_bin_mod), end_mod, edges_orig[0] ));
    DEBUG("update curbin mod %li: %g -> %g\n", std::distance(edges_mod, cur_bin_mod), *cur_bin_mod, *(cur_bin_mod+1));
  }
  auto i_bin_mod = cur_bin_mod - edges_mod;
  if( cur_bin_mod<end_mod ){
    // Find current bin's projection to the original range
    auto* cur_proj_to_orig = std::lower_bound( edges_orig, end_orig, *cur_bin_mod );
    DEBUG("found cur_proj %li: %g -> %g\n", std::distance(edges_orig, cur_proj_to_orig), *cur_proj_to_orig, *(cur_proj_to_orig+1));
    if (cur_proj_to_orig!=edges_orig) {
      cur_proj_to_orig = std::prev(cur_proj_to_orig);
      DEBUG("update cur_proj %li %g -> %g\n", std::distance(edges_orig, cur_proj_to_orig), *cur_proj_to_orig, *(cur_proj_to_orig+1));
    }
    auto i_proj = cur_proj_to_orig - edges_orig;
    if ( cur_proj_to_orig<end_orig && cur_bin_mod<end_mod ){
      auto* next_bin_mod = std::next(cur_bin_mod);
      // Iterate bins
      #ifdef DEBUG_ENL
      size_t iteration=0;
      #endif
      while( cur_bin_mod<end_mod && *cur_bin_mod<*end_orig && *next_bin_mod<m_range_max ){
        auto full_width = *next_bin_mod - *cur_bin_mod;

        auto* cur_edge{cur_bin_mod};
        #ifdef DEBUG_ENL
        bool cur_mod{true};
        #endif
        if(*cur_edge < *edges_orig){
          cur_edge = edges_orig;
          #ifdef DEBUG_ENL
          cur_mod=false;
          #endif
        }
        // Iterate inner bin edges
        while ( (cur_edge!=next_bin_mod) && (cur_proj_to_orig!=end_orig) ){
          auto* next_edge = std::min( next_bin_mod, std::next(cur_proj_to_orig), [](auto a, auto b){ return *a<*b; } );

          #ifdef DEBUG_ENL
          bool next_mod = next_edge==next_bin_mod;
          #endif

          double weight = ( *next_edge - *cur_edge )/full_width;

          #ifdef DEBUG_ENL
          if ( ((iteration++)%20)==0 ) {
          printf("\n%6s%14s%13s%15s%14s%8s %8s%8s%8s\n",
                "it",
                "curbin", "curproj", "curedge",
                "nextedge", "nextbin", "nextproj",
                "weight", "width");
          }
          printf("%6li"
                 "%7lij%6.3g""%7lii%6.3g"
                 "%7li%s%6.3g""%7li%s%6.3g"
                 "%8.3g%1s%8.3g""%8.3f%8.3g %s\n",
                 iteration,
                 i_bin_mod, *cur_bin_mod, i_proj, *cur_proj_to_orig,
                 std::distance(cur_mod?edges_mod:edges_orig, cur_edge), cur_mod?"j":"i", *cur_edge,
                 std::distance(next_mod?edges_mod:edges_orig, next_edge), next_mod?"j":"i", *next_edge,
                 *next_bin_mod, *next_bin_mod==*std::next(cur_proj_to_orig) ? "=":" ", *std::next(cur_proj_to_orig),
                 weight, full_width, weight==0.0 ? "*" : "" );
          #endif
          m_sparse_cache.insert(i_proj, i_bin_mod) = weight;

          cur_edge = next_edge;
          std::advance(cur_proj_to_orig, 1); i_proj++;
          #ifdef DEBUG_ENL
          cur_mod = next_mod;
          #endif
        }
        DEBUG("\n");

        if(*next_bin_mod!=*cur_proj_to_orig) {
          std::advance(cur_proj_to_orig, -1); i_proj--;
        }
        std::advance(cur_bin_mod, 1); i_bin_mod++;
        std::advance(next_bin_mod, 1);
      }
    }
  }
  DEBUG("\n");

  if ( m_propagate_matrix )
    fargs.rets[0].mat = m_sparse_cache;
}

