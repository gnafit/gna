#include "InSegment.hh"
#include "TypesFunctions.hh"

#define DEBUG_INSEGMENT

#include <algorithm>
#include <cmath>
using std::lower_bound;
using std::next;
using std::advance;
using std::prev;
using std::distance;
using std::fabs;

/**
 * @brief Constructor.
 */
InSegment::InSegment() {
  transformation_("segments")                                     /// Define transformation `segments`:
                                                                  ///   - with two inputs:
    .input("points")                                              ///     + `points` - fine x.
    .input("edges")                                               ///     + `edges` - coarse x bins.
                                                                  ///   - two outputs:
    .output("insegment")                                          ///     + relevant segment for each point.
    .output("widths")                                             ///     + output `width` with segment widths.
    .types(TypesFunctions::ifPoints<0>)                           ///   - `points` - is an array.
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)  ///   - `edges` is 1-dimensional array.
    .types(TypesFunctions::pass<0,0>)                             ///   - the dimension of the first output the same
                                                                  ///     as dimension of the first input (one index per input point).
    .types(TypesFunctions::edgesToBins<1,1>)                      ///   - the `widths` output has N-1 elements.
    .func(&InSegment::determineSegments)                          ///   - provide the function.
    ;
}

/**
 * @brief The transformation function.
 *
 * For each input points it detemines the relevant segment.
 */
void InSegment::determineSegments(FunctionArgs& fargs){
  double* insegment = fargs.rets[0].buffer;              /// The output array pointer.

  auto& edges_a = fargs.args[1].x;                       /// The edges data,
  auto  edge_first = edges_a.data();                     /// edges array pointer,
  auto  nedges=edges_a.size();
  auto  edge_last= next(edge_first, nedges-1);           /// the last edge.
  auto  edge_end = next(edge_first, nedges);             /// the end of the edges array.

  fargs.rets[1].x=edges_a.tail(nedges-1) - edges_a.head(nedges-1); /// Determine bin widths.

  auto& points_d=fargs.args[0];                          /// The points data,
  auto  point_first=points_d.buffer;                     /// points array pointer (current),
  auto  point_end=next(point_first, points_d.x.size());  /// points end pointer,
  auto  point=point_first;

  while(point<point_end){                                /// Iterate over all the points.
    if (*point<*edge_first-m_tolerance){                 /// Check if the point below the lower limit
      *insegment = -1;
    }
    else if (*point>=*edge_last){                        /// Check if the point is above the upper limit
      *insegment = static_cast<double>(nedges-1);
    }
    else{
      auto seg_next = lower_bound(edge_first, edge_end, *point); /// Find edge, that is greater or equal the current point.
      auto seg = prev(seg_next);                                 /// Fund the current bin
      if(seg_next<edge_end && fabs(*point-*seg_next)<m_tolerance){ /// If the point is below the next bin edge on less-then-tolerance
        seg=seg_next;                                            /// Assign the point to the next bin
      }
      *insegment = static_cast<double>(distance(edge_first, seg)); /// Store the data
    }

    #ifdef DEBUG_INSEGMENT
    printf("  % 4i point %8.4g -> segment %3i", (int)distance(point_first, point), *point, (int)*insegment);
    if( *insegment>=0 && *insegment<nedges-1 ){
      printf(": [%8.4g, %8.4g)", *next(edge_first, (int)*insegment), *next(edge_first, (int)*insegment+1));
    }
    printf("\n");
    #endif

    advance(point, 1);     /// Step to next point
    advance(insegment, 1); /// Step the write pointer
  }
}
