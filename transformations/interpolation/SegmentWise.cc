#include "SegmentWise.hh"
#include "TypesFunctions.hh"

//#define DEBUG_SEGMENTWISE

#include <algorithm>
#include <cmath>
using std::lower_bound;
using std::next;
using std::prev;
using std::distance;
using std::fabs;

/**
 * @brief Constructor.
 */
SegmentWise::SegmentWise() {
  transformation_("segments")                                     /// Define transformation `segments`:
                                                                  ///   - with two inputs:
    .input("points")                                              ///     + `points` - fine x.
    .input("edges")                                               ///     + `edges` - coarse x bins.
                                                                  ///   - two outputs:
    .output("segments")                                           ///     + output `segments` with segment indices.
    .output("widths")                                             ///     + output `width` with segment widths.
    .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>)  ///   - `points` is 1-dimensional array.
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)  ///   - `edges` is 1-dimensional array.
    .types(TypesFunctions::pass<1,0>)                             ///   - the dimension of the first output the same
                                                                  ///     as dimension of the second input (one index per edge).
    .types(TypesFunctions::edgesToBins<1,1>)                      ///   - the `widths` output has N-1 elements.
    .func(&SegmentWise::determineSegments)                        ///   - provide the function.
    ;
}

/**
 * @brief The transformation function.
 *
 * For each edge from the bin edges it find an index of the point, that is higher or equal to it.
 */
void SegmentWise::determineSegments(Args args, Rets rets){
  double* segment = rets[0].buffer;                      /// The output array pointer.

  auto& edges_a = args[1].x;                             /// The edges data,
  auto  edge = edges_a.data();                           /// edges array pointer,
  auto  nedges=edges_a.size();
  auto  edge_end = next(edge, nedges);                   /// the end of the edges array.

  rets[1].x=edges_a.tail(nedges-1) - edges_a.head(nedges-1); /// Determine bin widths.

  auto& points_d=args[0];                                /// The points data,
  auto  point_first=points_d.buffer;                     /// points array pointer,
  auto  point_end=next(point_first, points_d.x.size());  /// points end pointer,
  auto  point = point_first;                             /// current points pointer (to be iterated over).

  #ifdef DEBUG_SEGMENTWISE
  size_t edge_i{0};
  #endif
  while(edge<edge_end){                                  /// Iterate over all the edges.
    point = lower_bound(point, point_end, *edge);        /// Find point, that is greater or equal the current bin edge.
    auto point_i = distance(point_first, point);         /// Determine its index.

    if(point_i>0u){                                      /// Check if
      auto pp = prev(point);                             /// previous point
      if( fabs(*pp-*edge)<m_tolerance ){                 /// is below the bin edge on less than tolerance and
        point=pp;                                        /// assign it to the current bin if needed.
        point_i-=1u;
      }
    }
    *segment = static_cast<double>(point_i);             /// The output array is double: need to convert the index before writing.

    #ifdef DEBUG_SEGMENTWISE
    printf("%3i edge %8.4g, point %8.4g, idx %3i\n", (int)edge_i, *edge, *point, (int)point_i);
    edge_i++;
    #endif

    edge=next(edge);                                     /// Step to the next edge.
    segment=next(segment);                               /// Step the write pointer as well.
  }
}
