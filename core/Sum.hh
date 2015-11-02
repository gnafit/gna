#ifndef SUM_H
#define SUM_H

#include "GNAObject.hh"

class Sum: public GNASingleObject,
           public Transformation<Sum> {
public:
  Sum();

  void add(GNASingleObject &obj) {
    add(obj[0].outputs.single());
  }
  void add(const TransformationDescriptor &obj) {
    add(obj.outputs.single());
  }
  void add(const TransformationDescriptor::Outputs &outs) {
    add(outs.single());
  }
  void add(const OutputDescriptor &data);
  size_t size() const;
};

#endif // SUM_H
