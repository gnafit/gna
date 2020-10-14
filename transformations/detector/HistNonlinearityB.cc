#include "HistNonlinearityB.hh"
#include "TypesFunctions.hh"
#include <algorithm>
#include <fmt/format.h>

//#define DEBUG_ENLB

#ifdef DEBUG_ENLB
#define  DEBUG(...) do {                                      \
    printf(__VA_ARGS__);                                      \
} while (0);
#else
#define  DEBUG(...)
#endif

template <class F>
class final_act
{
public:
    explicit final_act(F f) noexcept
      : f_(std::move(f)), invoke_(true) {}

    final_act(final_act&& other) noexcept
     : f_(std::move(other.f_)),
       invoke_(other.invoke_)
    {
        other.invoke_ = false;
    }

    final_act(const final_act&) = delete;
    final_act& operator=(const final_act&) = delete;

    ~final_act() noexcept
    {
        if (invoke_) f_();
    }

private:
    F f_;
    bool invoke_;
};

HistNonlinearityB::HistNonlinearityB(bool propagate_matrix) :
HistSmearSparse(propagate_matrix)
{
    transformation_("matrix")
        .input("Edges", /*inactive*/true)
        .input("EdgesModified")
        .input("BackwardProjection")
        .output("FakeMatrix")
        .types(TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>)
        .types(TypesFunctions::ifBinsEdges<0,1>, TypesFunctions::ifSameShape2<1,2>)
        .types(TypesFunctions::toMatrix<0,0,0>)
        .types(&HistNonlinearityB::getEdges)
        .func(&HistNonlinearityB::calcMatrix);

    add_transformation();
    add_input();
    set_open_input();
}

void HistNonlinearityB::set(SingleOutput& bin_edges, SingleOutput& bin_centers_modified){
    if( m_initialized )
        throw std::runtime_error("HistNonlinearityB is already initialized");
    m_initialized = true;

    auto inputs = transformations.front().inputs;
    bin_edges.single()          >> inputs[0];
    bin_centers_modified.single() >> inputs[1];
}

void HistNonlinearityB::getEdges(TypesFunctionArgs& fargs) {
    m_edges = fargs.args[0].edges.data();
}

/**
 * @brief Calculate the conversion matrix
 * The algorithm:
 * 1. Find first X' (modified) bin above or equal X[0] (original)
 *
 */
void HistNonlinearityB::calcMatrix(FunctionArgs& fargs) {
    auto& args=fargs.args;

    size_t nbins  = args[0].arr.size();                        // Number of bins
    size_t nedges = nbins+1;                                   // Number of bin edges

    //
    // Iterators for the original edges and its projection
    //
    auto* orig_left               = m_edges;                   // Current left edge, original
    auto* orig_right              = orig_left + 1u;            // Current right edge, original
    auto* orig_proj_left          = args[1].arr.data();        // Projection of a current left edge / modified left edge
    auto* orig_proj_right         = orig_proj_left + 1u;       // Projection of a current right edge / modified right edge

    auto  orig_width              = *orig_right - *orig_left;  // Current bin width, original

    const auto  orig_lastedge_val = *(orig_left + nbins);      // Last original bin
    const auto* orig_start        = orig_left;                 // Start of the original edges
    const auto* orig_end          = orig_left + nedges;        // End of the original edges
    //const auto* orig_proj_end     = orig_proj_left + nedges;   // End of the projected edges

    //
    // Iterators for the modified edges and its backward projection on the unmodified axis
    //
    auto* mod_left                = orig_proj_left;            // Current left modified edge
    auto* mod_right               = orig_proj_left + 1u;       // Current right modified edge
    auto* mod_proj_left           = args[2].arr.data();        // Backward projection of a current modified left edge
    auto* mod_proj_right          = mod_proj_left + 1u;        // Backward projection of a current modified right edge

    const auto  mod_lastedge_val  = *(mod_left + nbins);       // Last modified bin edge
    const auto* mod_start         = mod_left;                  // Start of the modified edges
    const auto* mod_end           = mod_left + nedges;         // End of the modified edges
    //const auto* mod_proj_end      = mod_proj_left + nedges;    // End of the backward projected modified edges

    //
    // Lambda function to check range
    //
    auto check_range_orig = [&orig_right,
                       &orig_proj_left,
                       orig_end,
                       mod_lastedge_val, this]() -> bool
         {
             if(orig_right==orig_end) return false; // No more bins to iterate
             if(   *orig_proj_left >= mod_lastedge_val
                || *orig_proj_left >= this->m_range_max ) return false; // Projected bins are out of range

             return true; // Iteration successfull
         };

    auto check_range_mod = [mod_right,
                      mod_proj_left,
                      mod_end, orig_lastedge_val]() -> bool
         {
             if(mod_right==mod_end || mod_right==mod_end) return false; // No more bins to iterate
             if(*mod_proj_left>=orig_lastedge_val) return false;        // Projected bins are out of range

             return true; // Iteration successfull
         };

    //
    // Lambda function to step the current original and the current modified bin
    //
    auto step_orig = [&orig_left, &orig_right,
                      &orig_proj_left, &orig_proj_right,
                      &orig_width,
                      &check_range_orig]() -> bool
         {
             ++orig_left; ++orig_right;
             ++orig_proj_left; ++orig_proj_right;
             orig_width = *orig_right - *orig_left;

             return check_range_orig();
         };

    auto step_mod = [&mod_left, &mod_right,
                     &mod_proj_left, &mod_proj_right,
                     &check_range_mod]() -> bool
         {
             ++mod_left; ++mod_right;
             ++mod_proj_left; ++mod_proj_right;

             return check_range_mod();
         };

    //auto sync_mod_right = [mod_left, &mod_right,
                           //mod_start,
                           //&mod_proj_left, &mod_proj_right,
                           //&check_range_mod]() -> bool
         //{
             //mod_right = mod_left + 1u;

             //mod_proj_left += mod_left-mod_start;
             //mod_proj_right = mod_proj_left + 1u;

             //return check_range_mod();
         //};

    auto sync_orig_left = [&orig_left, orig_right,
                          orig_start,
                          &orig_proj_left, &orig_proj_right,
                          &check_range_orig]() -> bool
         {
             orig_left = orig_right - 1u;

             orig_proj_left += orig_left-orig_start;
             orig_proj_right = orig_proj_left + 1u;

             return check_range_orig();
         };

    auto sync_mod_left = [&mod_left, mod_right,
                          mod_start,
                          &mod_proj_left, &mod_proj_right,
                          &check_range_mod]() -> bool
         {
             mod_left = mod_right - 1u;

             mod_proj_left += mod_left-mod_start;
             mod_proj_right = mod_proj_left + 1u;

             return check_range_mod();
         };

    //
    // Initialize sparse matrix
    //
    m_sparse_cache.resize(nbins, nbins);
    m_sparse_cache.setZero();

    // Fill the matrix at exit
    final_act _f([&fargs, this](){
        if ( this->m_propagate_matrix )
            fargs.rets[0].mat = this->m_sparse_cache;
    });

    // 1. Set the range
    //     a. Find first right X' (modified) edge above orig_start (original)
    mod_right = std::upper_bound(mod_right, mod_end, *orig_start);
    if(!sync_mod_left()) return;

    //     b. Find first right X (original) edge above mod_proj_left (modified)
    orig_right = std::upper_bound(orig_right, orig_end, *mod_left);
    if(!sync_orig_left()) return;

    // 2. Start the iteration
    while(true){
        if(!step_orig()) return;
        if(!step_mod()) return;
    }

    //DEBUG("n=%zu, matrix n=%zu\n", bins+1, bins);

    //// Find the first bin center in the modified array that is above threshold
    //auto* cur_bin_center_mod = std::upper_bound(centers_mod, end_mod, m_range_min);
    //if(cur_bin_center_mod!=end_mod){
    //// Found a bin center within range
    //DEBUG("found cur_bin_center_mod %li: %g\n", std::distance(centers_mod, cur_bin_mod), *cur_bin_mod);

    //// Current bin center index
    //auto i_bin_mod = cur_bin_mod - centers_mod;
    //// Full 'original' width corresponding to the current bin
    //auto full_width = edges_orig[i_bin_mod+1] - edges_orig[i_bin_mod];
    //auto half_width = 0.5*half_width;
    //// Left and right modified edges
    //auto left_edge_mod  = *cur_bin_mod-half_width;
    //auto right_edge_mod = *cur_bin_mod+half_width;

    //// Find current bin's left/right edge projection to the original range
    //auto* left_proj_to_orig = std::lower_bound(edges_orig, end_orig, left_edge_mod);
    //auto* right_proj_to_orig = std::lower_bound(edges_orig, end_orig, right_edge_mod);

    //DEBUG("found cur_proj %li: %g -> %g\n", std::distance(edges_orig, cur_proj_to_orig), *cur_proj_to_orig, *(cur_proj_to_orig+1));
    //if (cur_proj_to_orig!=edges_orig) {
    //cur_proj_to_orig = std::prev(cur_proj_to_orig);
    //DEBUG("update cur_proj %li %g -> %g\n", std::distance(edges_orig, cur_proj_to_orig), *cur_proj_to_orig, *(cur_proj_to_orig+1));
    //}
    //// Index of the oritinal bin containing the current modified left edge
    //auto i_proj = cur_proj_to_orig - edges_orig;

    //if ( cur_proj_to_orig<end_orig ){
    //// The projection of the current center is below rightmost edge
    //// and current bin is within range
    //// Iterate bins
    //#ifdef DEBUG_ENLB
    //size_t iteration=0;
    //#endif
    //while( cur_bin_mod<end_mod && left_edge_mod<*end_orig && right_edge_mod<m_range_max ){
    //auto* cur_edge{cur_bin_mod};
    //#ifdef DEBUG_ENLB
    //bool cur_mod{true};
    //#endif
    //if(*cur_edge < *edges_orig){
    //cur_edge = edges_orig;
    //#ifdef DEBUG_ENLB
    //cur_mod=false;
    //#endif
    //}
    //// Iterate inner bin edges
    //while ( (cur_edge!=next_bin_mod) && (cur_proj_to_orig!=end_orig) ){
    //auto* next_edge = std::min( next_bin_mod, std::next(cur_proj_to_orig), [](auto a, auto b){ return *a<*b; } );

    //#ifdef DEBUG_ENLB
    //bool next_mod = next_edge==next_bin_mod;
    //#endif

    //double weight = ( *next_edge - *cur_edge )/full_width;

    //#ifdef DEBUG_ENLB
    //if ( ((iteration++)%20)==0 ) {
    //printf("\n%6s%14s%13s%15s%14s%8s %8s%8s%8s\n",
    //"it",
    //"curbin", "curproj", "curedge",
    //"nextedge", "nextbin", "nextproj",
    //"weight", "width");
    //}
    //printf("%6li"
    //"%7lij%6.3g""%7lii%6.3g"
    //"%7li%s%6.3g""%7li%s%6.3g"
    //"%8.3g%1s%8.3g""%8.3f%8.3g %s\n",
    //iteration,
    //i_bin_mod, *cur_bin_mod, i_proj, *cur_proj_to_orig,
    //std::distance(cur_mod?edges_mod:edges_orig, cur_edge), cur_mod?"j":"i", *cur_edge,
    //std::distance(next_mod?edges_mod:edges_orig, next_edge), next_mod?"j":"i", *next_edge,
    //*next_bin_mod, *next_bin_mod==*std::next(cur_proj_to_orig) ? "=":" ", *std::next(cur_proj_to_orig),
    //weight, full_width, weight==0.0 ? "*" : "" );
    //#endif
    //m_sparse_cache.insert(i_proj, i_bin_mod) = weight;

    //cur_edge = next_edge;
    //std::advance(cur_proj_to_orig, 1); i_proj++;
    //#ifdef DEBUG_ENLB
    //cur_mod = next_mod;
    //#endif
    //}
    //DEBUG("\n");

    //if(*next_bin_mod!=*cur_proj_to_orig) {
    //std::advance(cur_proj_to_orig, -1); i_proj--;
    //}
    //std::advance(cur_bin_mod, 1); i_bin_mod++;
    //std::advance(next_bin_mod, 1);
    //}
    //}
    //}     // cur_bin_center_mod!=end_mod
    //else{ // cur_bin_center_mod==end_mod
    //DEBUG("cur_bin_center_mod is outside\n");
    //}
    //DEBUG("\n");
}
