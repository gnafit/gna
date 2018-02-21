#include "SegmentWise.hh"

SegmentWise::SegmentWise(size_t nedges, const double* edges):
m_edges(Eigen::Map<const Eigen::ArrayXd>(edges, nedges))
{
	transformation_("segments")
		.output("edges")
		.finalize()
		;
}

void SegmentWise::defineTypes(Atypes args, Rtypes rets){
	rets[0]=DataType().points().view(m_edges);
}
