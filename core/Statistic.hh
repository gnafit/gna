#ifndef STATISTIC_H
#define STATISTIC_H

class Statistic {
public:
  virtual ~Statistic() { }
  virtual double value() = 0;
};

#endif // STATISTIC_H
