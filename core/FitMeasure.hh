#ifndef FITMEASURE_H
#define FITMEASURE_H

class FitMeasure {
public:
  virtual ~FitMeasure() { }
  virtual double value() = 0;
};

#endif // FITMEASURE_H
