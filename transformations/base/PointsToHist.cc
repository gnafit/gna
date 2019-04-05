#include "PointsToHist.hh"

PointsToHist::PointsToHist(SingleOutput& points, double left_edge) : m_left_edge(left_edge) {
    init(points);
}

PointsToHist::PointsToHist(SingleOutput& points){
    init(points);
}

void PointsToHist::init(SingleOutput& points) {
    size_t size_in = points.datatype().size();
    if(m_left_edge){
        m_edges.resize(size_in+1);
        m_edges[0] = m_left_edge.value();
        std::copy(points.data(), points.data()+size_in, m_edges.begin()+1);
    }
    else{
        m_edges.resize(size_in);
        std::copy(points.data(), points.data()+size_in, m_edges.begin());
    }
    m_fragile = fragile({points.single().getTaintflag()});
    transformation_("adapter")
        .input("points")
        .output("hist")
        .types([](PointsToHist* obj, TypesFunctionArgs targs){
                    targs.rets[0] = DataType().hist().edges(obj->m_edges);
                })
        .func([](PointsToHist* obj, FunctionArgs fargs) {
                fargs.args[0];
                auto& rets=fargs.rets;
                rets[0].x=0.0;
                rets.untaint();
                rets.freeze();
                })
        .finalize();

    points.single() >> transformations.front().inputs[0];
}
