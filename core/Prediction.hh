#ifndef PREDICTION_H
#define PREDICTION_H

#include "GNAObject.hh"

class Prediction: public GNASingleObject,
                  public Transformation<Prediction> {
public:
  Prediction();
  Prediction(const Prediction &other);

  Prediction &operator=(const Prediction &other);

  void append(SingleOutput &out);

  void calculateTypes(Atypes args, Rtypes rets);
  void calculatePrediction(Args args, Rets rets);

  size_t size() const;

  void update() const;
protected:
  Handle m_transform;
};

#endif // PREDICTION_H
