#ifndef DATA_H
#define DATA_H

#include <stddef.h>

#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <limits>

#include <Eigen/Dense>

enum class Status {
  Undefined = 0, Success, Failed,
};

enum class DataKind {
  Undefined = 0, Points, Hist,
};

struct DataType {
  template <typename T>
  class Points;
  Points<DataType> points();
  Points<const DataType> points() const;

  template <typename T>
  class Hist;
  Hist<DataType> hist();
  Hist<const DataType> hist() const;

  bool operator==(const DataType &other) const;
  bool operator!=(const DataType &other) const;

  void dump() const;

  bool defined() const { return kind != DataKind::Undefined; }
  static const DataType &undefined() {
    static DataType undefined;
    return undefined;
  }

  DataKind kind = DataKind::Undefined;
  size_t size;
  std::vector<double> edges = {};
  std::pair<double, double> bounds = {
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()
  };
};

template <typename T>
class DataType::Points {
  template <typename U>
  friend class DataType::Points;
public:
  Points(const Points<T> &other): m_type(other.m_type) { }
  Points(T &type): m_type(type) { }

  operator T() const { return m_type; }

  template <typename U>
  typename std::enable_if<std::is_convertible<U, T>::value, bool>::type
  operator==(const DataType::Points<U> &other) const {
    if (m_type.size != other.m_type.size) {
      return false;
    }
    return true;
  }

  void dump() const {
    fprintf(stderr, "size == %lu\n", size());
  }

  DataType::Points<T> &any() {
    return setKind();
  }
  bool valid() const {
    return m_type.kind == DataKind::Points;
  }

  DataType::Points<T> &size(size_t size) {
    m_type.size = size;
    return setKind();
  }
  size_t size() const {
    return m_type.size;
  }

  DataType::Points<T> &bounds(double min, double max) {
    m_type.bounds.first = min;
    m_type.bounds.second = max;
    return setKind();
  }
  DataType::Points<T> &bounds(const std::pair<double, double> &bounds) {
    m_type.bounds = bounds;
    return setKind();
  }
  const std::pair<double, double> &bounds() {
    return m_type.bounds;
  }

  DataType::Points<T> &min(double min) {
    m_type.bounds.first = min;
    return setKind();
  }
  double min() {
    return m_type.bounds.first;
  }

  DataType::Points<T> &max(double max) {
    m_type.bounds.second = max;
    return setKind();
  }
  double max() {
    return m_type.bounds.second;
  }
protected:
  DataType::Points<T> &setKind() {
    m_type.kind = DataKind::Points;
    return *this;
  }

  T &m_type;
};

inline DataType::Points<DataType> DataType::points() {
  return Points<DataType>(*this);
}

inline DataType::Points<const DataType> DataType::points() const {
  return Points<const DataType>(*this);
}

template <typename T>
class DataType::Hist {
  template <typename U>
  friend class DataType::Hist;
public:
  Hist(const Hist<T> &other): m_type(other.m_type) { }
  Hist(T &type): m_type(type) { }

  operator T() const { return m_type; }

  template <typename U>
  typename std::enable_if<std::is_convertible<U, T>::value, bool>::type
  operator==(const DataType::Hist<U> &other) const {
    if (m_type.size != other.m_type.size) {
      return false;
    }
    if (m_type.edges != other.m_type.edges) {
      return false;
    }
    return true;
  }

  void dump() const {
    fprintf(stderr, ", size == [%lu]", m_type.size);
    fprintf(stderr, ", Edges[%lu]", edges().size());
    fprintf(stderr, "\n");
  }

  DataType::Hist<T> &any() {
    return setKind();
  }
  bool valid() const {
    return m_type.kind == DataKind::Hist;
  }

  DataType::Hist<T> &bins(int nbins) {
    m_type.size = nbins;
    return setKind();
  }
  int bins() const {
    return m_type.size;
  }

  DataType::Hist<T> &edges(const std::vector<double> &edges) {
    m_type.edges = edges;
    return setKind();
  }
  template <typename U=T,
            typename = typename std::enable_if<!std::is_const<U>::value>::type>
  std::vector<double> &edges() {
    return m_type.edges;
  }
  const std::vector<double> &edges() const {
    return m_type.edges;
  }
protected:
  DataType::Hist<T> &setKind() {
    m_type.kind = DataKind::Hist;
    return *this;
  }

  T &m_type;
};

inline DataType::Hist<DataType> DataType::hist() {
  return Hist<DataType>(*this);
}

inline DataType::Hist<const DataType> DataType::hist() const {
  return Hist<const DataType>(*this);
}

inline bool DataType::operator==(const DataType &other) const {
  if (kind != other.kind) {
    return false;
  }
  switch (kind) {
  case DataKind::Points:
    return points() == other.points();
  case DataKind::Hist:
    return hist() == other.hist();
  default:
    return false;
  }
  return true;
}

inline bool DataType::operator!=(const DataType &other) const {
  return !(*this == other);
}

inline void DataType::dump() const {
  fprintf(stderr, "DataType, ");
  fprintf(stderr, "Kind == ");
  switch (kind) {
  case DataKind::Undefined:
    fprintf(stderr, "Undefined\n");
    break;
  case DataKind::Points:
    fprintf(stderr, "Points");
    points().dump();
    break;
  case DataKind::Hist:
    fprintf(stderr, "Hist");
    hist().dump();
    break;
  default:
    fprintf(stderr, "INVALID VALUE!\n");
    break;
  }
}

template <typename T>
class Data {
public:
  Data(const DataType &dt)
    : type(dt), status(Status::Undefined)
  {
    allocate();
  }

  bool defined() const { return type.defined(); }
  void allocate();

  const DataType type;
  Eigen::Array<T, Eigen::Dynamic, 1> x;
  Status status;
};

template <typename T>
inline void Data<T>::allocate() {
  if (type.kind == DataKind::Undefined) {
    return;
  }
  x.resize(type.size);
}

#endif // DATA_H
