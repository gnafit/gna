#include "SegmentWise.hh"
#include "TypesFunctions.hh"

#include <algorithm>
using std::lower_bound;

SegmentWise::SegmentWise() {
	transformation_("segments")
		.input("points")
		.input("edges")
		.output("segments")
		.types(TypesFunctions::ifPoints<0>, TypesFunctions::ifPoints<1>)
		.types(TypesFunctions::ifNd<0,1>,   TypesFunctions::ifNd<1,1>)
		.types(TypesFunctions::pass<1,0>)
		.types(&SegmentWise::defineTypes)
		.func(&SegmentWise::determineSegments)
		;
}

void SegmentWise::defineTypes(Atypes args, Rtypes rets){
	rets[1] = rets[0]; // NOTE: the buffer pointer is not passed via assignment as it should be
}

void SegmentWise::determineSegments(Args args, Rets rets){
  //cur_bin = std::prev(std::lower_bound( std::next(cur_bin), end_mod, edges_orig[0] ));
}
