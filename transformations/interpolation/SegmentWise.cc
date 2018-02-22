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

SegmentWise::SegmentWise() {
  transformation_("segments")
    .input("points")
    .input("edges")
    .output("segments")
    .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>)
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)
    .types(TypesFunctions::pass<1,0>)
    .func(&SegmentWise::determineSegments)
    ;
}

void SegmentWise::determineSegments(Args args, Rets rets){
  double* segment = rets[0].buffer;

  auto& edges_d = args[1];
  auto  edge = args[1].buffer;
  auto  edge_end = next(edge, edges_d.x.size());

  auto& points_d=args[0];
  auto  point_first=points_d.buffer;
  auto  point_end=next(point_first, points_d.x.size());
  auto  point = point_first;

  #ifdef DEBUG_SEGMENTWISE
  size_t edge_i{0};
  #endif
  while(edge<edge_end){
    point = lower_bound(point, point_end, *edge);
    auto point_i = distance(point_first, point);

    if(point_i>0u){
      auto pp = prev(point);
      if( fabs(*pp-*edge)<m_tolerance ){
        point=pp;
        point_i-=1u;
      }
    }
    *segment = static_cast<double>(point_i);

    #ifdef DEBUG_SEGMENTWISE
    printf("%3i edge %8.4g, point %8.4g, idx %3i\n", (int)edge_i, *edge, *point, (int)point_i);
    edge_i++;
    #endif

    edge=next(edge);
    segment=next(segment);
  }
}
