#include "PointsToHist.hh"

PointsToHist::PointsToHist(SingleOutput& points, double left_most_bin) {
    std::vector<double> edges{left_most_bin};
    edges.resize(points.datatype().size() + 1);
    std::copy(points.data(), points.data() + points.datatype().size(), edges.begin()+1);

    transformation_("adapter")
        .input("points")
        .output("hist")
        .types([edges](PointsToHist* obj, TypesFunctionArgs targs){
                targs.rets[0] = DataType().hist().edges(edges);
                })
        .func([](PointsToHist* obj, FunctionArgs fargs) {
                fargs.rets[0].x = fargs.args[0].x;
                })
        .finalize();

    auto inputs = transformations.front().inputs;
    points.single() >> inputs[0];
}
