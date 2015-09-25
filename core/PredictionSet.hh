#ifndef PREDICTIONSET_H
#define PREDICTIONSET_H

#include <vector>
#include <list>

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

  const Data<const double> &view() const;
protected:
  Status build(Arguments args, Returns rets);
  Status buildTypes(ArgumentTypes args, ReturnTypes rets);

  Handle m_transform;

  ClassDef(PredictionSet, 1);
};

#endif // PREDICTIONSET_H
