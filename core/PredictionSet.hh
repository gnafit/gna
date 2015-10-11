#ifndef PREDICTIONSET_H
#define PREDICTIONSET_H

#include "GNAObject.hh"

class PredictionSet: public GNAObject,
                     public Transformation<PredictionSet> {
public:
  TransformationDef(PredictionSet)
  PredictionSet();
  PredictionSet(const PredictionSet &other);

  PredictionSet &operator=(const PredictionSet &other);

  void append(GNASingleObject &obj) {
    append(obj[0].outputs.single());
  }
  void append(const TransformationDescriptor &obj) {
    append(obj.outputs.single());
  }
  void append(const TransformationDescriptor::Outputs &outs) {
    append(outs.single());
  }
  void append(const OutputDescriptor &data);
  size_t size() const;

  void update() const;
  const double *data() const;
protected:
  Handle m_transform;
};

#endif // PREDICTIONSET_H
