#ifndef PREDICTION_H
#define PREDICTION_H

#include "GNAObject.hh"

class Prediction: public GNASingleObject,
                  public Transformation<Prediction> {
public:
  Prediction();
  Prediction(const Prediction &other);

  Prediction &operator=(const Prediction &other);

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

#endif // PREDICTION_H
