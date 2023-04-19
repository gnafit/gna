#include "HistNonlinearityD.hh"
#include "TypesFunctions.hh"
#include <algorithm>
#include <fmt/format.h>
#include "final_act.hh"

using std::max;
using GNA::DataPropagation;
using GNA::MatrixType;

// #define DEBUG_ENLD

HistNonlinearityD::HistNonlinearityD(GNA::DataPropagation propagate_matrix) :
HistSmearSparse(propagate_matrix, MatrixType::Any)
{
    transformation_("matrix")
        .input("EdgesIn", /*inactive*/true)
        .input("EdgesInModified")
        .input("EdgesOut", /*inactive*/true)
        .input("EdgesOutBackwardProjection")
        .output("FakeMatrix")
        .types(TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>)
        .types(TypesFunctions::ifHist<2>, TypesFunctions::if1d<2>)
        .types(TypesFunctions::ifBinsEdges<0,1>)
        .types(TypesFunctions::ifBinsEdges<2,3>)
        .types(TypesFunctions::toMatrix<2,0,0>)
        .types(&HistNonlinearityD::getEdges)
        .func(&HistNonlinearityD::calcMatrix);

    add_transformation();
    add_input();
    set_open_input();
}

void HistNonlinearityD::set(SingleOutput& orig, SingleOutput& orig_proj, SingleOutput& mod, SingleOutput& mod_proj){
    if( m_initialized )
        throw std::runtime_error("HistNonlinearityD is already initialized");
    m_initialized = true;

    auto inputs = transformations.front().inputs;
    orig.single()      >> inputs[0];
    orig_proj.single() >> inputs[1];
    mod.single()       >> inputs[2];
    mod_proj.single()  >> inputs[3];
}

void HistNonlinearityD::getEdges(TypesFunctionArgs& fargs) {
    auto& args = fargs.args;
    if (args[0].defined()){
        m_edges_in = fargs.args[0].edges.data();
    }
    if (args[2].defined()){
        m_edges_out = fargs.args[2].edges.data();
    }
}

#ifdef DEBUG_ENLD
#define  DEBUG(...) do {                                      \
    printf(__VA_ARGS__);                                      \
} while (0);
#define DEBUG_ITERATION()                                     \
        printf(                                               \
              "iter "                                         \
              "orig %3zu %7.3f->%7.3f "                       \
              "proj %7.3f->%7.3f "                            \
              "mod %3zu %7.3f->%7.3f "                        \
              "mod_proj %7.3f->%7.3f "                        \
              "\n"                                            \
              ,                                               \
              orig_idx, *orig_left,      *orig_right,         \
                        *orig_proj_left, *orig_proj_right,    \
              mod_idx,  *mod_left,       *mod_right,          \
                        *mod_proj_left,  *mod_proj_right      \
              );

#define DEBUG_STEP_ORIG()                                     \
        printf(                                               \
              "step "                                         \
              "orig %3zu %7.3f->%7.3f "                       \
              "proj %7.3f->%7.3f "                            \
              "width %7.3f "                                  \
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
              "step "                                         \
              "                                            "  \
              "mod %3zu %7.3f->%7.3f "                        \
              "mod_proj %7.3f->%7.3f "                        \
              "to end %td "                                   \
              "\n"                                            \
              ,                                               \
              mod_idx,  *mod_left,       *mod_right,          \
                        *mod_proj_left,  *mod_proj_right,     \
                         mod_end-mod_right                    \
              );
#define DEBUG_WEIGHT()                                        \
        printf(                                               \
              "current  %7.3f->%7.3f weight %7.3f "           \
              "exit orig %i exit mod %i"                      \
              "\n"                                            \
              ,                                               \
              left, right, weight,                            \
              exit_orig, exit_mod                             \
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
 */
void HistNonlinearityD::calcMatrix(FunctionArgs& fargs) {
    auto& args=fargs.args;

    size_t nbins_in  = args[0].arr.size();                     // Number of bins (in)
    size_t nedges_in = nbins_in+1;                             // Number of bin edges (in)
    size_t nbins_out  = args[2].arr.size();                    // Number of bins (out)
    size_t nedges_out = nbins_out+1;                           // Number of bin edges (out)

    //
    // Iterators for the original edges and its projection
    //
    size_t orig_idx               = 0u;                        // Current original bin index
    auto* orig_left               = m_edges_in;                // Current left edge, original
    auto* orig_right              = orig_left + 1u;            // Current right edge, original
    auto* orig_proj_left          = args[1].arr.data();        // Projection of a current left edge / modified left edge
    auto* orig_proj_right         = orig_proj_left + 1u;       // Projection of a current right edge / modified right edge

    auto  orig_width              = *orig_right - *orig_left;  // Current bin width, original

    const auto  orig_lastedge_val = *(orig_left + nbins_in);      // Last original bin
    const auto* orig_start        = orig_left;                 // Start of the original edges
    const auto* orig_end          = orig_left + nedges_in;        // End of the original edges
    //const auto* orig_proj_end     = orig_proj_left + nedges_in;   // End of the projected edges

    //
    // Iterators for the modified edges and its backward projection on the unmodified axis
    //
    size_t mod_idx                = 0u;                        // Current modified bin index
    auto* mod_left                = m_edges_out;               // Current left modified edge
    auto* mod_right               = mod_left + 1u;             // Current right modified edge
    auto* mod_proj_left           = args[3].arr.data();        // Backward projection of a current modified left edge
    auto* mod_proj_right          = mod_proj_left + 1u;        // Backward projection of a current modified right edge

    const auto  mod_lastedge_val  = *(mod_left + nbins_out);   // Last modified bin edge
    const auto* mod_start         = mod_left;                  // Start of the modified edges
    const auto* mod_end           = mod_left + nedges_out;     // End of the modified edges
    const auto* mod_proj_start    = mod_proj_left;             // End of the backward projected modified edges
    const auto* mod_proj_end      = mod_proj_left + nedges_out;// End of the backward projected modified edges

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
                            mod_end, orig_lastedge_val,
                            this]() -> bool
         {
             if(mod_right==mod_end) return true; // Exit: no more bins to iterate
             if(*mod_proj_left>=orig_lastedge_val) return true; // Exit: projected bins are out of range
             if(*mod_right>=this->m_range_max) return true; // Exit: touched the range

             return false; // Iteration successfull, no need to exit
         };

    //
    // Lambda function to step the current original and the current modified bin
    //
    auto step_orig = [&orig_idx, &orig_left, &orig_right,
                      &orig_proj_left, &orig_proj_right,
                      &orig_width,
                      &check_range_orig
#ifdef DEBUG_ENLD
                      , orig_end
#endif
    ]() -> bool
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
                     &check_range_mod
#ifdef DEBUG_ENLD
                     , mod_end
#endif
                    ]() -> bool
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

    auto sync_mod_right = [&mod_idx, &mod_left, &mod_right,
                          mod_start,
                          &mod_proj_left, &mod_proj_right,
                          &check_range_mod]() -> bool
         {
             mod_right = mod_left + 1u;
             mod_idx   = mod_left-mod_start;

             mod_proj_left += mod_left-mod_start;
             mod_proj_right = mod_proj_left + 1u;

             return check_range_mod();
         };

    auto sync_mod_proj_left = [&mod_idx, &mod_left, &mod_right,
                               mod_start, mod_proj_start,
                               &mod_proj_left, &mod_proj_right,
                               &check_range_mod]() -> bool
         {
             mod_proj_left = mod_proj_right - 1u;
             mod_idx       = mod_proj_left-mod_proj_start;

             mod_left  = mod_start + mod_idx;
             mod_right = mod_left + 1u;

             return check_range_mod();
         };

    //
    // Initialize sparse matrix
    //
    m_sparse_cache.resize(nbins_out, nbins_in);
    m_sparse_cache.setZero();

    // Fill the matrix at return
    final_act _f([&fargs, this](){
        this->m_sparse_cache.makeCompressed();
        if ( this->m_propagate_matrix )
            fargs.rets[0].mat = this->m_sparse_cache;
    });

    // 1. Set the range
    DEBUG("Initial\n");
    DEBUG_ITERATION();

    //     a. Find first left X' (modified) edge above m_range_min
    if(*mod_left <= m_range_min){
        DEBUG("Update mod (range)\n");
        mod_left = std::upper_bound(mod_left, mod_end-1, m_range_min);
        if(sync_mod_right()) return;
        DEBUG_ITERATION();
    }

    //     b. Find first right X' (modified) edge above mod_start (modified)
    if(*mod_right <= *mod_start){
        DEBUG("Update mod\n");
        mod_right = std::upper_bound(mod_right, mod_end, *mod_start);
        if(sync_mod_left()) return;
        DEBUG_ITERATION();
    }

    //     c. Find first right X (projected back) edge above orig_start (original)
    if(*mod_proj_right <= *orig_start){
        DEBUG("Update mod_proj\n");
        mod_proj_right = std::upper_bound(mod_proj_right, mod_proj_end, *orig_start);
        if(sync_mod_proj_left()) return;
        DEBUG_ITERATION();
    }

    //     d. Find first right X (original) edge above mod_proj_left (modified)
    if(*orig_right <= *mod_proj_left) {
        DEBUG("Update orig\n");
        orig_right = std::upper_bound(orig_right, orig_end, *mod_proj_left);
        if(sync_orig_left()) return;
        DEBUG_ITERATION();
    }

    DEBUG("Start iteration at %zu, %zu of %zu (in), %zu (out)\n", orig_idx, mod_idx, nbins_in, nbins_out);
    // 2. Start the iteration
    while(orig_idx<nbins_in && mod_idx<nbins_out){
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
        if(need_step_mod && !exit_mod) {
            exit_mod=step_mod();
            made_step=true;
        }

        if(exit_orig&&exit_mod) return;

        if(!made_step){
            throw std::runtime_error("HistNonlinearityD unable to make a step. Should not happen.");
        }
    }
}
