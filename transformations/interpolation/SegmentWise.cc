#include "SegmentWise.hh"

#include <algorithm>
using std::lower_bound;

SegmentWise::SegmentWise(size_t nedges, const double* edges):
m_edges(Eigen::Map<const Eigen::ArrayXd>(edges, nedges))
{
	transformation_("segments")
		.input("points")
		.output("segments")
		.output("edges")
		//.types(&Atypes::ifPoints<0>, Atypes::ifNd<0,1>)
		.types(&SegmentWise::defineTypes)
		.func(&SegmentWise::determineSegments)
		;
}

void SegmentWise::defineTypes(Atypes args, Rtypes rets){
	rets[0].points().view(m_edges);
	rets[1] = rets[0]; // NOTE: the buffer pointer is not passed via assignment as it should be
}

void SegmentWise::determineSegments(Args args, Rets rets){
  //cur_bin = std::prev(std::lower_bound( std::next(cur_bin), end_mod, edges_orig[0] ));
}
