#ifndef DATA_H
#define DATA_H

#include <stddef.h>

#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <limits>

enum class Status {
  Undefined = 0, Success, Failed,
};

enum class DataKind {
  Undefined = 0, Scalar, Series, Hist,
};

enum class ShapeType {
  Undefined = 0, None = 0, Any, Fixed,
};

struct DataType {
  template <typename T>
  class Numeric;

  template <typename T>
  class Scalar;
  Scalar<DataType> scalar();
  Scalar<const DataType> scalar() const;

  template <typename T>
  class Series;
  Series<DataType> series();
  Series<const DataType> series() const;

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
  ShapeType shapetype = ShapeType::Undefined;
  std::vector<int> shape = {};
  std::vector<double> edges = {};
  std::pair<double, double> bounds = {
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()
  };
};

template <typename T>
class DataType::Scalar {
  template <typename U>
  friend class DataType::Scalar;
public:
  Scalar(const Scalar<T> &other): m_type(other.m_type) { }
  Scalar(T &type): m_type(type) { }

  operator T() const { return m_type; }

  template <typename U>
  typename std::enable_if<std::is_convertible<U, T>::value, bool>::type
  operator==(const DataType::Scalar<U> &/* other */) const {
    return true;
  }

  void dump() const {
    fprintf(stderr, "\n");
  }

  DataType::Scalar<T> &any() {
    return setKind();
  }
  bool valid() const {
    return m_type.kind == DataKind::Scalar;
  }

  DataType::Scalar<T> &bounds(double min, double max) {
    m_type.bounds.first = min;
    m_type.bounds.second = max;
    return setKind();
  }
  DataType::Scalar<T> &bounds(const std::pair<double, double> &bounds) {
    m_type.bounds = bounds;
    return setKind();
  }
  const std::pair<double, double> &bounds() {
    return m_type.bounds;
  }

  DataType::Scalar<T> &min(double min) {
    m_type.bounds.first = min;
    return setKind();
  }
  double min() {
    return m_type.bounds.first;
  }

  DataType::Scalar<T> &max(double max) {
    m_type.bounds.second = max;
    return setKind();
  }
  double max() {
    return m_type.bounds.second;
  }
protected:
  DataType::Scalar<T> &setKind() {
    m_type.kind = DataKind::Scalar;
    return *this;
  }

  T &m_type;
};

inline DataType::Scalar<DataType> DataType::scalar() {
  return Scalar<DataType>(*this);
}

inline DataType::Scalar<const DataType> DataType::scalar() const {
  return Scalar<const DataType>(*this);
}

template <typename T>
class DataType::Series {
  template <typename U>
  friend class DataType::Series;
public:
  Series(const Series<T> &other): m_type(other.m_type) { }
  Series(T &type): m_type(type) { }

  operator T() const { return m_type; }

  template <typename U>
  typename std::enable_if<std::is_convertible<U, T>::value, bool>::type
  operator==(const DataType::Series<U> &other) const {
    if (m_type.shapetype != other.m_type.shapetype) {
      return false;
    }
    if (m_type.shape != other.m_type.shape) {
      return false;
    }
    return true;
  }

  void dump() const {
    fprintf(stderr, ", Shape[%lu] == {", shape().size());
    for (auto s: shape()) {
      fprintf(stderr, "%d", s);
    }
    fprintf(stderr, "}\n");
  }

  DataType::Series<T> &any() {
    m_type.shapetype = ShapeType::Any;
    return setKind();
  }
  bool valid() const {
    return m_type.kind == DataKind::Series;
  }

  DataType::Series<T> &shape(const std::vector<int> &shape) {
    m_type.shape = shape;
    return setKind();
  }
  std::vector<int> &shape() {
    return m_type.shape;
  }
  const std::vector<int> &shape() const {
    return m_type.shape;
  }
protected:
  DataType::Series<T> &setKind() {
    m_type.kind = DataKind::Series;
    return *this;
  }

  T &m_type;
};

inline DataType::Series<DataType> DataType::series() {
  return Series<DataType>(*this);
}

inline DataType::Series<const DataType> DataType::series() const {
  return Series<const DataType>(*this);
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
    if (m_type.shapetype != other.m_type.shapetype) {
      return false;
    }
    if (m_type.shape != other.m_type.shape) {
      return false;
    }
    if (m_type.edges != other.m_type.edges) {
      return false;
    }
    return true;
  }

  void dump() const {
    fprintf(stderr, ", Shape[%lu]", m_type.shape.size());
    fprintf(stderr, ", Edges[%lu]", edges().size());
    fprintf(stderr, "\n");
  }

  DataType::Hist<T> &any() {
    m_type.shapetype = ShapeType::Any;
    return setKind();
  }
  bool valid() const {
    return m_type.kind == DataKind::Hist;
  }
  bool concrete() const {
    return valid() && m_type.shapetype == ShapeType::Fixed;
  }

  DataType::Hist<T> &bins(int nbins) {
    m_type.shapetype = ShapeType::Fixed;
    m_type.shape.resize(1);
    m_type.shape[0] = nbins;
    return setKind();
  }
  int bins() const {
    return m_type.shape[0];
  }

  DataType::Hist<T> &edges(const std::vector<double> &edges) {
    m_type.edges = edges;
    return setKind();
  }
  template <typename U=T,
            typename =typename std::enable_if<!std::is_const<U>::value>::type>
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
  case DataKind::Scalar:
    fprintf(stderr, "Scalar");
    scalar().dump();
    break;
  case DataKind::Series:
    fprintf(stderr, "Series");
    series().dump();
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
  typedef std::vector<typename std::remove_const<T>::type> StorageType;
public:
  Data()
    : storage(new StorageType), buffer(nullptr),
      size(0), status(Status::Undefined)
    { }
  Data(const Data<T> &other)
    : type(other.type), storage(copyStorage(other)),
      buffer(&(*storage)[0]), size(other.size), status(other.status)
    { }
  template <typename U = typename std::add_const<T>::type>
  Data(const Data<U> &other)
    : type(other.type), storage(copyStorage(other)),
      buffer(&(*storage)[0]), size(other.size), status(other.status)
    { }
  Data(Data<T> &&other)
    : type(other.type), storage(std::move(other.storage)),
      buffer(&(*storage)[0]), size(other.size), status(other.status)
    { }
  template <typename U = typename std::add_const<T>::type>
  Data(Data<U> &&other)
    : type(other.type), storage(std::move(other.storage)),
      buffer(&(*storage)[0]), size(other.size), status(other.status)
    { }
  Data<T> &operator=(const Data<T> &other) {
    return assign(other);
  }
  template <typename U = typename std::add_const<T>::type>
  Data<T> &operator=(const Data<U> &other) {
    return assign(other);
  }

  Data<const T> &view() {
    return *reinterpret_cast<Data<const T>*>(this);
  }
  const Data<const T> &view() const {
    return *reinterpret_cast<const Data<const T>*>(this);
  }

  bool defined() const { return type.defined(); }
  void allocate();

  template <typename U,
            typename K = typename std::remove_const<T>::type,
            typename L = typename std::remove_const<U>::type,
            typename = typename std::enable_if<std::is_same<T, K>::value>::type,
            typename = typename std::enable_if<std::is_same<K, L>::value>::type>
  void setContent(const Data<U> &other) const {
    std::copy(other.storage->begin(), other.storage->end(), storage->begin());
  }

  DataType type;
  std::unique_ptr<StorageType> storage;
  T *buffer;
  size_t size;
  Status status;
private:
  template <typename U>
  static std::unique_ptr<StorageType> copyStorage(const Data<U> &other) {
    std::unique_ptr<StorageType> storage;
    storage.reset(new StorageType(*other.storage));
    return storage;
  }

  template <typename U>
  Data<T> &assign(const Data<U> &other) {
    type = other.type;
    storage = std::move(copyStorage(other));
    if (storage->size() > 0) {
      buffer = &(*storage)[0];
    } else {
      buffer = nullptr;
    }
    size = other.size;
    status = other.status;
    return *this;
  }
};

template <typename T>
inline void Data<T>::allocate() {
  if (type.kind == DataKind::Undefined) {
    return;
  }
  size = std::accumulate(type.shape.begin(), type.shape.end(), 1,
                            std::multiplies<size_t>());
  if (size > storage->size()) {
    storage->resize(size);
    buffer = &(*storage)[0];
  }
}

#endif // DATA_H
