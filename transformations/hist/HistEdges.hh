#pragma once

#include <vector>

#include "GNAObject.hh"
#include "TypesFunctions.hh"

class HistEdges: public GNASingleObject,
                 public TransformationBind<HistEdges> {
public:
  HistEdges() {
    transformation_("histedges")
      .input("hist")
      .output("edges")
      .types(TypesFunctions::ifHist<0>,
             [](TypesFunctionArgs& fargs) {
               fargs.rets[0] = DataType().points().shape(fargs.args[0].edges.size());
             })
      .func([](FunctionArgs& fargs) {
              auto& edges = fargs.args[0].type.edges;
              fargs.rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&edges[0], edges.size());
              fargs.rets.freeze();
            });
  };

  HistEdges(SingleOutput& out) : HistEdges() {
    t_[0].inputs()[0].connect(out.single());
  }
};
