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
             [](Atypes args, Rtypes rets) {
             rets[0] = DataType().points().shape(args[0].edges.size());
             })
      .func([](Args args, Rets rets) {
            auto& edges = args[0].type.edges;
            rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&edges[0], edges.size());
            rets.freeze();
            });
  };
};