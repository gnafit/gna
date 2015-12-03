#ifndef DATA_H
#define DATA_H

#include <stddef.h>

#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <limits>
#include <stdexcept>

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

  size_t size() const {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<size_t>());
  }

  bool preallocated() { return buffer != nullptr; }
  void preallocated(double *buf) { buffer = buf; }

  DataKind kind = DataKind::Undefined;
  std::vector<size_t> shape;

  std::vector<double> edges = {};
  std::pair<double, double> bounds = {
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()
  };

  double *buffer = nullptr;
};

template <typename T>
class DataType::Points {
  template <typename U>
  friend class DataType::Points;
public:
  Points(const Points<T> &other): m_type(other.m_type) { }
  Points(T &type): m_type(type) { }

  operator T() const { return m_type; }

  bool operator==(const DataType::Points<T> &other) const {
    if (m_type.shape != other.m_type.shape) {
      return false;
    }
    return true;
  }

  void dump() const {
    switch (shape().size()) {
    case 0:
      fprintf(stderr, "shape == empty\n");
      break;
    case 1:
      fprintf(stderr, "shape == (%lu)\n", shape()[0]);
      break;
    case 2:
      fprintf(stderr, "shape == (%lu, %lu)\n", shape()[0], shape()[1]);
      break;
    default:
      fprintf(stderr, "shape == WTF(%lu)?\n", shape().size());
      assert(shape().size() < 3);
      break;
    }
  }

  DataType::Points<T> &any() {
    return setKind();
  }
  bool valid() const {
    return m_type.kind == DataKind::Points;
  }

  DataType::Points<T> &shape(size_t shape0) {
    m_type.shape = std::vector<size_t>{shape0};
    return setKind();
  }
  DataType::Points<T> &shape(size_t shape0, size_t shape1) {
    m_type.shape = std::vector<size_t>{shape0, shape1};
    return setKind();
  }
  const std::vector<size_t> &shape() const {
    return m_type.shape;
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

  bool operator==(const DataType::Hist<T> &other) const {
    if (m_type.shape != other.m_type.shape) {
      return false;
    }
    if (m_type.edges != other.m_type.edges) {
      return false;
    }
    return true;
  }

  void dump() const {
    fprintf(stderr, ", bins == [%lu]", m_type.shape[0]);
    fprintf(stderr, ", Edges[%lu]", edges().size());
    fprintf(stderr, "\n");
  }

  DataType::Hist<T> &any() {
    return setKind();
  }
  bool valid() const {
    return m_type.kind == DataKind::Hist;
  }

  DataType::Hist<T> &bins(size_t nbins) {
    m_type.shape = std::vector<size_t>{nbins};
    return setKind();
  }
  int bins() const {
    return m_type.shape[0];
  }

  DataType::Hist<T> &edges(const std::vector<double> &edges) {
    m_type.edges = edges;
    return bins(edges.size()-1);
  }
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
  typedef Eigen::Array<T, Eigen::Dynamic, 1> ArrayXT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

  typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ArrayXXT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;
public:
  Data(const DataType &dt, T *buf)
    : type(dt)
  {
    if (!buf) {
      allocated.reset(new T[dt.size()]);
      buf = allocated.get();
    }
    buffer = buf;
    if (dt.shape.size() == 1) {
      new (&arr) Eigen::Map<ArrayXT>(buf, dt.shape[0]);
      new (&vec) Eigen::Map<VectorXT>(buf, dt.shape[0]);
    } else if (dt.shape.size() == 2) {
      new (&arr) Eigen::Map<ArrayXT>(buf, dt.shape[0]*dt.shape[1]);
      new (&vec) Eigen::Map<VectorXT>(buf, dt.shape[0]*dt.shape[1]);

      new (&arr2d) Eigen::Map<ArrayXXT>(buf, dt.shape[0], dt.shape[1]);
      new (&mat) Eigen::Map<MatrixXT>(buf, dt.shape[0], dt.shape[1]);
    }
  }

  const DataType type;
  Status state{Status::Undefined};

  T *buffer{nullptr};
  std::unique_ptr<T> allocated{nullptr};

  Eigen::Map<ArrayXT> arr{nullptr, 0};
  Eigen::Map<VectorXT> vec{nullptr, 0};

  Eigen::Map<ArrayXXT> arr2d{nullptr, 0, 0};
  Eigen::Map<MatrixXT> mat{nullptr, 0, 0};

  Eigen::Map<ArrayXT> &x = arr;
};

#endif // DATA_H
