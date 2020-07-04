#pragma once

#include <vector>

#include "GNAObject.hh"
#include "TypesFunctions.hh"

class HistEdges: public GNASingleObject,
                 public TransformationBind<HistEdges> {
public:
  HistEdges() {
    transformation_("histedges")
      .input("hist", /*inactive*/true)
      .output("edges")
      .output("centers")
      .output("widths")
      .types(TypesFunctions::ifHist<0>,
             [](TypesFunctionArgs& fargs) {
               auto nedges = fargs.args[0].edges.size();
               auto nbins  = nedges-1;
               auto& rets = fargs.rets;
               rets[0] = DataType().points().shape(nedges);
               rets[1] = DataType().points().shape(nbins);
               rets[2] = DataType().points().shape(nbins);
             })
      .func([](FunctionArgs& fargs) {
              auto& input   = fargs.args[0].type.edges;
              auto& rets    = fargs.rets;
              auto& edges   = rets[0].x;
              auto& centers = rets[1].x;
              auto& widths  = rets[2].x;
              auto nbins = input.size()-1;
              edges   = Eigen::Map<const Eigen::ArrayXd>(&input[0], input.size());
              centers = 0.5*(edges.head(nbins)+edges.tail(nbins));
              widths  = edges.tail(nbins)-edges.head(nbins);
              rets.untaint();
              rets.freeze();
            });
  };

  HistEdges(SingleOutput& out) : HistEdges() {
    t_[0].inputs()[0].connect(out.single());
  }
};
