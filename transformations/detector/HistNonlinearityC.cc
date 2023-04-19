#include "HistNonlinearityC.hh"
#include "TypesFunctions.hh"
#include <algorithm>
#include <fmt/format.h>
#include <stdexcept>
#include "final_act.hh"

// #define DEBUG_ENLC
using GNA::DataPropagation;
using GNA::MatrixType;

HistNonlinearityC::HistNonlinearityC(float nbins_factor, size_t nbins_extra, DataPropagation propagate_matrix) :
HistSmearSparse(propagate_matrix, MatrixType::Any),
m_nbins_factor(nbins_factor),
m_nbins_extra(nbins_extra)
{
    if (nbins_factor<1.0 || nbins_factor>10.0){
        throw std::domain_error("Invalid nbins_factor: expect 1<f<10");
    }
    transformation_("matrix")
        .input("Edges", /*inactive*/true)
        .input("EdgesModified")
        .input("BackwardProjection")
        .output("FakeMatrix")
        .output("OutEdges")
        .types(TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>)
        .types(TypesFunctions::ifBinsEdges<0,1>, TypesFunctions::ifSameShape2<1,2>)
        .types(&HistNonlinearityC::types)
        .func(&HistNonlinearityC::calcMatrix);

    add_transformation();
    add_input();
    set_open_input();
}

void HistNonlinearityC::set(SingleOutput& orig, SingleOutput& orig_proj, SingleOutput& mod_proj){
    if( m_initialized )
        throw std::runtime_error("HistNonlinearityC is already initialized");
    m_initialized = true;

    auto inputs = transformations.front().inputs;
    orig.single()      >> inputs[0];
    orig_proj.single() >> inputs[1];
    mod_proj.single()  >> inputs[2];
}

void HistNonlinearityC::types(TypesFunctionArgs& fargs) {
    auto& arg0 = fargs.args[0];
    if(!arg0.defined()){
        return;
    }
    m_edges = arg0.edges.data();

    auto nbins = arg0.size();
    auto nbins_new = static_cast<size_t>(nbins*m_nbins_factor) + m_nbins_extra;
    fargs.rets[0].points().shape(nbins_new, nbins);
    fargs.rets[1].points().shape(nbins_new+1);
}

#ifdef DEBUG_ENLC
#define  DEBUG(...) do {                                      \
    printf(__VA_ARGS__);                                      \
} while (0);
#define DEBUG_ITERATION()                                     \
        printf(                                               \
              "iter "                                         \
              "orig %3zu %7.5f->%7.5f "                       \
              "proj %7.5f->%7.5f "                            \
              "mod %3zu %7.5f->%7.5f "                        \
              "mod_proj %7.5f->%7.5f "                        \
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
              "orig %3zu %7.5f->%7.5f "                       \
              "proj %7.5f->%7.5f "                            \
              "width %7.5f "                                  \
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
              "                                                "  \
              "mod %3zu %7.5f->%7.5f "                        \
              "mod_proj %7.5f->%7.5f "                        \
              "to end %td "                                   \
              "\n"                                            \
              ,                                               \
              mod_idx,  *mod_left,       *mod_right,          \
                        *mod_proj_left,  *mod_proj_right,     \
                         mod_end-mod_right                    \
              );
#define DEBUG_WEIGHT()                                        \
        printf(                                               \
              "current  %7.5f->%7.5f weight %7.5f "           \
              "\n"                                            \
              ,                                               \
              left, right, weight                             \
              );
#define DEBUG_EXIT()                                          \
        printf(                                               \
              "exit orig %i exit mod %i"                      \
              "\n"                                            \
              ,                                               \
              exit_orig, exit_mod                             \
              );
#define DEBUG_GLOBAL(loc)                                     \
        printf(                                               \
              "global %3zu edge %7.5f %s"                     \
              "\n"                                            \
              ,                                               \
              global_idx, *output_left,                       \
              loc                                             \
              );
#else
#define  DEBUG(...)
#define  DEBUG_ITERATION()
#define  DEBUG_STEP_ORIG()
#define  DEBUG_STEP_MOD()
#define  DEBUG_WEIGHT()
#define  DEBUG_EXIT()
#define  DEBUG_GLOBAL(loc)
#endif

/**
 * @brief Calculate the conversion matrix
 */
void HistNonlinearityC::calcMatrix(FunctionArgs& fargs) {
    auto& args=fargs.args;
    auto& rets=fargs.rets;

    size_t nbins  = args[0].arr.size();                        // Number of bins
    size_t nedges = nbins+1;                                   // Number of bin edges


    //
    // Iteration over the target edges
    //
    size_t nedges_out = rets[1].type.size();                   // Number of output bin edges
    size_t nbins_out  = nedges_out-1;                          // Number of output bins
    size_t global_idx = 0u;
    auto* output_left = rets[1].arr.data();                    // Current output edge pointer
    const auto* output_end = output_left+nedges_out;           // End of the output edges

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
    const auto* mod_proj_start    = mod_proj_left;             // End of the backward projected modified edges
    const auto* mod_proj_end      = mod_proj_left + nedges;    // End of the backward projected modified edges

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
#ifdef DEBUG_ENLC
                      , &orig_end
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
#ifdef DEBUG_ENLC
                     , &mod_end
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
    // Initialize sparse and rebin matrices
    //
    m_sparse_cache.resize(nbins_out, nbins);
    m_sparse_cache.setZero();
    m_sparse_cache.reserve(nbins_out);

    // m_sparse_rebin.resize(nbins, nbins_out);
    // m_sparse_rebin.setZero();
    // m_sparse_rebin.reserve(nbins_out);

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

    DEBUG("Start iteration at %zu, %zu of %zu\n", orig_idx, mod_idx, nbins);
    // 2. Start the iteration
    double proj_left{0.0};
    while(orig_idx<nbins && mod_idx<nbins){
        DEBUG_ITERATION();

        //
        // Determine the current interval
        //
        double left;
        if(*orig_left>*mod_proj_left){
            left = *orig_left;
            proj_left = *orig_proj_left;
        }
        else{
            left = *mod_proj_left;
            proj_left = *mod_left;
        }
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
        m_sparse_cache.insert(global_idx, orig_idx)=weight;
        // m_sparse_rebin.insert(mod_idx, global_idx)=1.0;

        DEBUG_WEIGHT();

        *output_left = proj_left;
        DEBUG_GLOBAL("loop");
        global_idx++;
        output_left++;

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
        DEBUG_EXIT();

        //
        // Exception in case output is overflowed
        //
        if(output_left>=output_end){
            throw fargs.rets.error("HistNonlinearityC: output overflow");
        }

        //
        // Stop iteration if nothing is left
        //
        if(exit_orig&&exit_mod) {
            break;
        }

        //
        // Exception in case  next step is undefined
        //
        if(!made_step){
            throw fargs.rets.error("HistNonlinearityC unable to make a step. Should not happen.");
        }
    }

    // Set the last edge
    *output_left = *mod_left;
    //
    // Fill missing edges with dummy data. Assume that elements are zeros
    //
    DEBUG_GLOBAL("last edge");
    output_left++;
    for(; output_left<output_end; output_left++){
        proj_left+=m_overflow_step;
        *output_left = proj_left;
        DEBUG_GLOBAL("overflow");
    }

    auto* edge_ptr=rets[1].arr.data();
    auto edge_prev = *edge_ptr;
    edge_ptr++;
    for(; edge_ptr<output_end; edge_ptr++){
        auto edge_new = *edge_ptr;
        if (edge_new<edge_prev){
            throw fargs.rets.error(fmt::format("HistNonlinearityC: edges are not aligned (..., {0}, {1}, ...). Perhaps inputs are out of sync.", edge_prev, edge_new));
        }
        edge_prev = edge_new;
    }
}
