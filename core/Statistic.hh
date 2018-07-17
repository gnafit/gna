#pragma once

class Statistic {
public:
  virtual ~Statistic() { }
  virtual double value() = 0;
};
