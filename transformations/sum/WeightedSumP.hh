#pragma once

#include <string>
#include <vector>

#include "GNAObject.hh"

class WeightedSumP: public GNASingleObject,
                    public TransformationBind<WeightedSumP> {
public:
  WeightedSumP();
  WeightedSumP(const OutputDescriptor::OutputDescriptors& outputs);

  void add(SingleOutput &a, SingleOutput &b);

protected:
  void sum(FunctionArgs& fargs);
  void check(TypesFunctionArgs& fargs);
  void add(SingleOutput &a);
};
