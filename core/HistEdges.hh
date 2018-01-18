#ifndef HISTEDGES_H
#define HISTEDGES_H

#include <vector>

#include "GNAObject.hh"

class HistEdges: public GNASingleObject,
                 public Transformation<HistEdges> {
public:
  HistEdges() {
    transformation_(this, "histedges")
      .input("hist")
      .output("edges")
      .types(Atypes::ifHist<0>,
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

#endif // HISTEDGES_H
