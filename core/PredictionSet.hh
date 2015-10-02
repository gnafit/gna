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

  void add(const OutputDescriptor &data);
  size_t size() const;

  void update() const;
  const double *data() const;
protected:
  Handle m_transform;

  ClassDef(PredictionSet, 1);
};

#endif // PREDICTIONSET_H
