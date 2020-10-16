#include "HistNonlinearityB.hh"
#include "TypesFunctions.hh"
#include <algorithm>
#include <fmt/format.h>
#include "final_act.hh"

using std::min;
using std::max;

#define DEBUG_ENLB

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

void HistNonlinearityB::set(SingleOutput& orig, SingleOutput& orig_proj, SingleOutput& mod_proj){
    if( m_initialized )
        throw std::runtime_error("HistNonlinearityB is already initialized");
    m_initialized = true;

    auto inputs = transformations.front().inputs;
    orig.single()      >> inputs[0];
    orig_proj.single() >> inputs[1];
    mod_proj.single()  >> inputs[2];
}

void HistNonlinearityB::getEdges(TypesFunctionArgs& fargs) {
    m_edges = fargs.args[0].edges.data();
}

#ifdef DEBUG_ENLB
#define  DEBUG(...) do {                                      \
    printf(__VA_ARGS__);                                      \
} while (0);
#define DEBUG_ITERATION()                                     \
        printf(                                               \
              "orig %3zu %6.2f->%6.2f "                       \
              "proj %6.2f->%6.2f "                            \
              "mod %3zu %6.2f->%6.2f "                        \
              "mod_proj %6.2f->%6.2f "                        \
              "\n"                                            \
              ,                                               \
              orig_idx, *orig_left,      *orig_right,         \
                        *orig_proj_left, *orig_proj_right,    \
              mod_idx,  *mod_left,       *mod_right,          \
                        *mod_proj_left,  *mod_proj_right      \
              );

#define DEBUG_STEP_ORIG()                                     \
        printf(                                               \
              "orig %3zu %6.2f->%6.2f "                       \
              "proj %6.2f->%6.2f "                            \
              "width %6.2f "                                  \
              "to end %td "                                   \
              "\n"                                            \
              ,                                               \
              orig_idx, *orig_left,      *orig_right,         \
                        *orig_proj_left, *orig_proj_right,    \
                         orig_width,                          \
                         orig_end-orig_right                  \
              );

#define DEBUG_STEP_MOD()                                      \
        printf(                                               \
              "                                            "  \
              "mod %3zu %6.2f->%6.2f "                        \
              "mod_proj %6.2f->%6.2f "                        \
              "to end %td "                                   \
              "\n"                                            \
              ,                                               \
              mod_idx,  *mod_left,       *mod_right,          \
                        *mod_proj_left,  *mod_proj_right,     \
                         mod_end-mod_right                    \
              );
#define DEBUG_WEIGHT()                                        \
        printf(                                               \
              "current  %6.2f->%6.2f weight %6.2f "            \
              "exit orig %i exit mod %i"                      \
              "\n"                                            \
              ,                                               \
              left, right, weight, exit_orig, exit_mod        \
              );
#else
#define  DEBUG(...)
#define  DEBUG_ITERATION()
#define  DEBUG_STEP_ORIG()
#define  DEBUG_STEP_MOD()
#define  DEBUG_WEIGHT()
#endif

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
    size_t orig_idx               = 0u;                        // Current original bin index
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
    size_t mod_idx                = 0u;                        // Current modified bin index (same binning as original)
    auto* mod_left                = orig_left;                 // Current left modified edge (same binning as original)
    auto* mod_right               = orig_left + 1u;            // Current right modified edge
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
             if(orig_right==orig_end) return true; // Exit: no more bins to iterate
             if(   *orig_proj_left >= mod_lastedge_val
                || *orig_proj_left >= this->m_range_max ) return true; // Exit: projected bins are out of range

             return false; // Iteration successfull, no need to exit
         };

    auto check_range_mod = [&mod_right,
                            &mod_proj_left,
                            mod_end, orig_lastedge_val]() -> bool
         {
             if(mod_right==mod_end) return true; // Exit: no more bins to iterate
             if(*mod_proj_left>=orig_lastedge_val) return true;        // Exit: projected bins are out of range

             return false; // Iteration successfull, no need to exit
         };

    //
    // Lambda function to step the current original and the current modified bin
    //
    auto step_orig = [&orig_idx, &orig_left, &orig_right,
                      &orig_proj_left, &orig_proj_right,
                      &orig_width,
                      &check_range_orig, orig_end]() -> bool
         {
             ++orig_idx;
             ++orig_left; ++orig_right;
             ++orig_proj_left; ++orig_proj_right;
             orig_width = *orig_right - *orig_left;

             DEBUG_STEP_ORIG();

             return check_range_orig();
         };

    auto step_mod = [&mod_idx, &mod_left, &mod_right,
                     &mod_proj_left, &mod_proj_right,
                     &check_range_mod, mod_end]() -> bool
         {
             ++mod_idx;
             ++mod_left; ++mod_right;
             ++mod_proj_left; ++mod_proj_right;

             DEBUG_STEP_MOD();

             return check_range_mod();
         };

    auto sync_orig_left = [&orig_idx, &orig_left, &orig_right,
                           orig_start,
                           &orig_proj_left, &orig_proj_right,
                           &check_range_orig]() -> bool
         {
             orig_left = orig_right - 1u;
             orig_idx  = orig_left-orig_start;

             orig_proj_left += orig_idx;
             orig_proj_right = orig_proj_left + 1u;

             return check_range_orig();
         };

    auto sync_mod_left = [&mod_idx, &mod_left, &mod_right,
                          mod_start,
                          &mod_proj_left, &mod_proj_right,
                          &check_range_mod]() -> bool
         {
             mod_left = mod_right - 1u;
             mod_idx  = mod_left-mod_start;

             mod_proj_left += mod_left-mod_start;
             mod_proj_right = mod_proj_left + 1u;

             return check_range_mod();
         };

    //
    // Initialize sparse matrix
    //
    m_sparse_cache.resize(nbins, nbins);
    m_sparse_cache.setZero();

    // Fill the matrix at return
    final_act _f([&fargs, this](){
        if ( this->m_propagate_matrix )
            fargs.rets[0].mat = this->m_sparse_cache;
    });

    // 1. Set the range
    //     a. Find first right X' (modified) edge above orig_start (original)
    DEBUG("Initial\n");
    DEBUG_ITERATION();
    mod_right = std::upper_bound(mod_right, mod_end, *orig_start);
    if(sync_mod_left()) return;

    DEBUG("Update mod\n");
    DEBUG_ITERATION();

    //     b. Find first right X (original) edge above mod_proj_left (modified)
    orig_right = std::upper_bound(orig_right, orig_end, *mod_proj_left);
    if(sync_orig_left()) return;

    DEBUG("Update orig\n");
    DEBUG_ITERATION();

    DEBUG("Start iteration at %zu, %zu of %zu\n", orig_idx, mod_idx, nbins);
    // 2. Start the iteration
    while(orig_idx<nbins && mod_idx<nbins){
        DEBUG_ITERATION();

        //
        // Determine the current interval
        //
        double left = max(*orig_left, *mod_proj_left);
        double right;
        bool need_step_orig{false}, need_step_mod{false};
        bool exit_orig{false}, exit_mod{false};
        if(*orig_right>*mod_proj_right){
            right = *mod_proj_right;
            need_step_orig = false;
            need_step_mod  = true;
        }
        else{
            right = *orig_right;
            need_step_orig = true;
            need_step_mod  = *orig_right==*mod_proj_right;
        }

        //
        // Compute the weight
        //
        auto weight = (right-left)/orig_width;
        m_sparse_cache.insert(mod_idx, orig_idx)=weight;

        DEBUG_WEIGHT();

        //
        // Make next step and/or exit
        //
        bool made_step = false;
        if(need_step_orig && !exit_orig) {
            exit_orig=step_orig();
            made_step=true;
        }
        if(need_step_mod  && !exit_mod) {
            exit_mod=step_mod();
            made_step=true;
        }

        if(exit_orig&&exit_mod) return;

        if(!made_step){
            throw std::runtime_error("HistNonlinearityB unable to make a step. Should not happen.");
        }
    }
}
