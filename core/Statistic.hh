#pragma once

class Statistic {
public:
  virtual ~Statistic() = default;
  virtual double value() = 0;
};
