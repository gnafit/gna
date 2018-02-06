#ifndef DATA_H
#define DATA_H

#include <stddef.h>

#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <type_traits>

#include <Eigen/Dense>
#include "config_vars.h"

#ifdef GNA_CUDA_SUPPORT 
#include "GNAcuGpuArray.hh"
#include "GNAcuDataLocation.hh"
#include "cuda_config_vars.h"
#endif // GNA_CUDA_SUPPORT

/**
 * @brief Data status flag.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
enum class Status {
  Undefined = 0,  ///< Initial value, undefined.
  Success,        ///< Data is OK.
  Failed,         ///< Data evaluation failed.
};

/**
 * @brief Data type (kind) flag.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
enum class DataKind {
  Undefined = 0,  ///< Initial value, undefined.
  Points,         ///< Points, for any dimensional arrays.
  Hist,           ///< Hist, for any dimensional histograms.
};

/**
 * @brief Data type specification.
 *
 * Defines the dimensions, sizes, kind (array/histogram).
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
struct DataType {
  /** @brief Points (array) DataType initializer class (CRTP). Castable to DataType. */
  template <typename T>
  class Points;
  Points<DataType> points();                                    ///< Points (array) DataType initialization and configuration.
  Points<const DataType> points() const;                        ///< Points (array) DataType initialization and configuration.

  /** @brief Hist DataType initializer class (CRTP). Castable to DataType. */
  template <typename T>
  class Hist;
  Hist<DataType> hist();                                        ///< Hist DataType initialization and configuration.
  Hist<const DataType> hist() const;                            ///< Hist DataType initialization and configuration.

  bool operator==(const DataType &other) const;                 ///< Check if data types are identical.
  bool operator!=(const DataType &other) const;                 ///< Check if data types are not identical.

  void dump() const;

  bool defined() const { return kind != DataKind::Undefined; }  ///< Check if DataType is initialized and has specific DataKind.
  /**
   * @brief Return the undefined DataType instance (static).
   * @return static undefined DataType.
   */
  static const DataType &undefined() {
    static DataType undefined;
    return undefined;
  }

  /**
   * @brief Compute the size of a buffer (number of elements).
   * The size is computed by multiplying numbers of elements in all dimensions.
   * @return data size.
   */
  size_t size() const {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<size_t>());
  }

  bool preallocated() const { return buffer != nullptr; }       ///< Check if data buffer is preallocated and not managed by Data.

  /**
   * @brief Set the preallocated (double) buffer pointer.
   *
   * The preallocated buffer should be consistent with datatype.
   *
   * @param buf -- preallocated double buffer.
   *
   * @todo Preallocated buffer is explicitly double, while the Data<> is a template with arbitrary datatype. Make DataType<T> a template?
   */
  void preallocated(double *buf) { buffer = buf; }

  DataKind kind = DataKind::Undefined;                         ///< DataKind: points (array) or histogram?
  std::vector<size_t> shape;                                   ///< Data dimensions.

  std::vector<double> edges = {};                              ///< Bin edges for 1D histogram. @todo Support 2D histograms.
  //std::pair<double, double> bounds = {
    //-std::numeric_limits<double>::infinity(),
     //std::numeric_limits<double>::infinity()
  //};

  double *buffer = nullptr;                                    ///< Preallocated data buffer (double).
};

/**
 * @brief DataType initializer and configurator for DataKind=Points (CRTP).
 *
 * DataType::Points enables chain DataType initialization for DataKind=Points.
 * After initialization DataType::Points may be casted to DataType.
 *
 * Usage example:
 * ```cpp
 * auto dt1=DataType().points().shape(size1);
 * auto dt2=DataType().points().shape(size1, size2);
 * ```
 *
 * @tparam T -- data type DataType::Points is castable to.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
template <typename T>
class DataType::Points {
  template <typename U>
  friend class DataType::Points;
public:
  /**
   * @brief Clone constructor.
   * @param other -- other instance of DataType::Points to clone DataType from.
   */
  Points(const Points<T> &other): m_type(other.m_type) { }

  /**
   * @brief Clone constructor.
   * @param type -- DataType instance to initialize from.
   */
  Points(T &type): m_type(type) { }

  operator T() const { return m_type; }                            ///< Cast to (return) the initialized DataType instance.

  /**
   * @brief Compare the dimensions with the dimensions of another instance of DataType::Points.
   * @param other -- another instance of DataType::Points.
   */
  bool operator==(const DataType::Points<T> &other) const {
    if (m_type.shape != other.m_type.shape) {
      return false;
    }
    return true;
  }

  /**
   * @brief Prints to stderr the dimensions of the data.
   * Undefined for the case of number of dimensions is higher than 2.
   */
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

  /**
   * @brief Change DataKind to DataKind::Points.
   * Useful, when DataType::Points instance was initialized with DataType of the other kind.
   * @return `*this`.
   */
  DataType::Points<T> &any() {
    return setKind();
  }
  /**
   * @brief Check that DataKind is valid.
   * Useful, when DataType::Points instance was initialized with DataType of the other kind.
   */
  bool valid() const {
    return m_type.kind == DataKind::Points;
  }

  /**
   * @brief Set the data size and dimensions for 1D data.
   * @param shape0 -- first dimension size.
   * @return `*this`.
   */
  DataType::Points<T> &shape(size_t shape0) {
    m_type.shape = std::vector<size_t>{shape0};
    return setKind();
  }

  /**
   * @brief Set the data size and dimensions for 2D data.
   * @param shape0 -- first dimension size.
   * @param shape1 -- second dimension size.
   * @return `*this`.
   */
  DataType::Points<T> &shape(size_t shape0, size_t shape1) {
    m_type.shape = std::vector<size_t>{shape0, shape1};
    return setKind();
  }
  /**
   * @brief Set the data size and dimensions for multidimensional data.
   * @param shape -- vector with dimensions.
   * @return `*this`.
   */
  DataType::Points<T> &shape(std::vector<size_t> shapes) {
    m_type.shape = shapes;
    return setKind();
  }
  /**
   * @brief Return the dimensions.
   * @return vector with dimensions.
   */
  const std::vector<size_t> &shape() const {
    return m_type.shape;
  }

  //DataType::Points<T> &bounds(double min, double max) {
    //m_type.bounds.first = min;
    //m_type.bounds.second = max;
    //return setKind();
  //}
  //DataType::Points<T> &bounds(const std::pair<double, double> &bounds) {
    //m_type.bounds = bounds;
    //return setKind();
  //}
  //const std::pair<double, double> &bounds() {
    //return m_type.bounds;
  //}

  //DataType::Points<T> &min(double min) {
    //m_type.bounds.first = min;
    //return setKind();
  //}
  //double min() {
    //return m_type.bounds.first;
  //}

  //DataType::Points<T> &max(double max) {
    //m_type.bounds.second = max;
    //return setKind();
  //}
  //double max() {
    //return m_type.bounds.second;
  //}
protected:
  /**
   * @brief Set the DataKind to DataKind::Points.
   * @return `*this`.
   */
  DataType::Points<T> &setKind() {
    m_type.kind = DataKind::Points;
    return *this;
  }

  T &m_type;                                                       ///< DataType instance initialized within DataType::Points.
};

/**
 * @brief Points (array) DataType initialization.
 * @return DataType CRTP initializer for DataKind=Points.
 */
inline DataType::Points<DataType> DataType::points() {
  return Points<DataType>(*this);
}

/** @copydoc DataType::points() */
inline DataType::Points<const DataType> DataType::points() const {
  return Points<const DataType>(*this);
}

/**
 * @brief DataType initializer for DataKind=Hist (CRTP).
 *
 * DataType::Hist enables chain DataType initialization for DataKind=Hist.
 * After initialization DataType::Hist may be casted to DataType.
 *
 * Usage example:
 * ```cpp
 * auto dt1 = DataType().hist().edges(edges_vec);
 * auto dt2 = DataType().hist().edges(n, edges);
 * ```
 *
 * @tparam T -- data type DataType::Hist is castable to.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
template <typename T>
class DataType::Hist {
  template <typename U>
  friend class DataType::Hist;
public:
  /**
   * @brief Clone constructor.
   * @param other -- other instance of DataType::Hist to clone DataType from.
   */
  Hist(const Hist<T> &other): m_type(other.m_type) { }
  /**
   * @brief Clone constructor.
   * @param type -- DataType instance to initialize from.
   */
  Hist(T &type): m_type(type) { }

  operator T() const { return m_type; }                            ///< Cast to (return) the initialized DataType instance.

  /**
   * @brief Compare the dimensions and bin edges with data of another instance of DataType::Hist.
   * @param other -- another instance of DataType::Hist.
   */
  bool operator==(const DataType::Hist<T> &other) const {
    if (m_type.shape != other.m_type.shape) {
      return false;
    }
    if (m_type.edges != other.m_type.edges) {
      return false;
    }
    return true;
  }

  /**
   * @brief Prints to stderr the dimensions and bin edges of the data.
   */
  void dump() const {
    fprintf(stderr, ", bins == [%lu]", m_type.shape[0]);
    fprintf(stderr, ", Edges[%lu]", edges().size());
    fprintf(stderr, "\n");
  }

  /**
   * @brief Change DataKind to DataKind::Hist.
   * Useful, when DataType::Hist instance was initialized with DataType of the other kind.
   * @return `*this`.
   */
  DataType::Hist<T> &any() {
    return setKind();
  }
  /**
   * @brief Check that DataKind is valid.
   * Useful, when DataType::Hist instance was initialized with DataType of the other kind.
   */
  bool valid() const {
    return m_type.kind == DataKind::Hist;
  }

  /**
   * @brief Set the number of bins.
   * @param nbins -- the number of bins.
   * @return `*this`.
   */
  DataType::Hist<T> &bins(size_t nbins) {
    m_type.shape = std::vector<size_t>{nbins};
    return setKind();
  }

  /**
   * @brief Get the number of bins.
   * @return number of bins.
   */
  size_t bins() const {
    return m_type.shape[0];
  }

  /**
   * @brief Set the bin edges via std::vector.
   * @param edges -- vector with bin edges.
   * @return `*this`.
   */
  DataType::Hist<T> &edges(const std::vector<double> &edges) {
    m_type.edges = edges;
    return bins(edges.size()-1);
  }

  /**
   * @brief Set the bin edges via double buffer.
   * @param n -- number of bin edges.
   * @param edges -- buffer with bin edges.
   * @return `*this`.
   */
  DataType::Hist<T> &edges(size_t n, double* edges) {
    m_type.edges.assign(edges, edges+n);
    return bins(n-1);
  }

  /**
   * @brief Get the vector with bin edges.
   * @return edges.
   */
  std::vector<double> &edges() {
    return m_type.edges;
  }

  /** @copydoc DataType::Hist::edges() */
  const std::vector<double> &edges() const {
    return m_type.edges;
  }
protected:
  /**
   * @brief Set the DataKind to DataKind::Hist.
   * @return `*this`.
   */
  DataType::Hist<T> &setKind() {
    m_type.kind = DataKind::Hist;
    return *this;
  }

  T &m_type;                                                       ///< DataType instance initialized within DataType::Hist.
};

/**
 * @brief Hist DataType initialization and configuration.
 * @return DataType CRTP initializer for DataKind=Hist.
 */
inline DataType::Hist<DataType> DataType::hist() {
  return Hist<DataType>(*this);
}

/** @copydoc DataType::hist() */
inline DataType::Hist<const DataType> DataType::hist() const {
  return Hist<const DataType>(*this);
}

/**
 * @brief Check if data types are identical.
 *
 * Compares that DataKind values. If needed uses DataType::Hist::operator==() or DataType::Points::operator==()
 * for further comparison.
 */
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

/**
 * @brief Check if data types are not identical.
 * @copydetails DataType::operator==()
 */
inline bool DataType::operator!=(const DataType &other) const {
  return !(*this == other);
}

/**
 * @brief Dumps the DataType contents to stderr.
 *
 * - Dumps the DataKind.
 * - Dumps the other contents via DataType::Points::dump() or DataType::Hist::dump().
 */
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

/**
 * @brief Generic data class.
 *
 * The class holds the following informations:
 *   - The buffer of date of type T (usually double).
 *   - DatType specification, i.e. buffer size, dimensions, bin edges.
 *   - Several views on the buffer via Eigen classes:
 *     * 1- and 2- dimensinal arrays.
 *     * Vector and Matrix.
 *
 *
 * @tparam T -- data type to hold buffer for (double).
 * @author Dmitry Taychenachev
 * @date 2015
 */
template <typename T>
class Data {
  typedef Eigen::Array<T, Eigen::Dynamic, 1> ArrayXT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

  typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ArrayXXT;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;
public:
  /**
   * @brief Constructor.
   * @param dt -- DataType specification.
   *
   * Constructor does:
   *   - if dt has preallocated buffer uses it is to store the data. The user must ensure that the buffer size is consistent with DataType requirements.
   *   - allocates the buffer, enough to hold data, specified for the DataType.
   *   - Initializes array, vector and matrix views on the buffer.
   *
   * @exception std::runtime_error in case DataType is not defined.
   * @exception std::bad_typeid in case preallocated buffer type is not the same as Data<T> type.
   */
  Data(const DataType &dt)
    : type(dt)
  {
    if(!dt.defined()){
      throw std::runtime_error("Using undefined DataType to initialize data");
    }
    if (dt.preallocated()) {
      if(!std::is_same<T*, decltype(dt.buffer)>::value) {
        throw std::bad_typeid();
      }
      this->buffer = dt.buffer;
    }
    else {
      allocated.reset(new T[dt.size()]);
      this->buffer = allocated.get();
    }
    if (dt.shape.size() == 1) {
      new (&this->arr)   Eigen::Map<ArrayXT>(  this->buffer, dt.shape[0] );
      new (&this->vec)   Eigen::Map<VectorXT>( this->buffer, dt.shape[0] );
    } else if (dt.shape.size() == 2) {
      new (&this->arr)   Eigen::Map<ArrayXT>(  this->buffer, dt.shape[0]*dt.shape[1] );
      new (&this->vec)   Eigen::Map<VectorXT>( this->buffer, dt.shape[0]*dt.shape[1] );

      new (&this->arr2d) Eigen::Map<ArrayXXT>( this->buffer, dt.shape[0], dt.shape[1] );
      new (&this->mat)   Eigen::Map<MatrixXT>( this->buffer, dt.shape[0], dt.shape[1] );
    }
  }
#ifdef GNA_CUDA_SUPPORT
  DataLocation require_gpu();
#endif // GNA_CUDA_SUPPORT

  const DataType type;                             ///< data type.
  Status state{Status::Undefined};                 ///< data status.

  T *buffer{nullptr};                              ///< the buffer. Data ownership is undefined.
  std::unique_ptr<T> allocated{nullptr};           ///< the buffer initialized within Data. Deallocates the data when destructed.

  Eigen::Map<ArrayXT> arr{nullptr, 0};             ///< 1D array view.
  Eigen::Map<VectorXT> vec{nullptr, 0};            ///< 1D vector view.

  Eigen::Map<ArrayXXT> arr2d{nullptr, 0, 0};       ///< 2D array view.
  Eigen::Map<MatrixXT> mat{nullptr, 0, 0};         ///< 2D matrix view.

  Eigen::Map<ArrayXT> &x = arr;                    ///< 1D array view shorthand.

#ifdef GNA_CUDA_SUPPORT
  std::unique_ptr<GNAcuGpuArray<T>> gpuArr{nullptr};
#endif // GNA_CUDA_SUPPORT
}; 

#ifdef GNA_CUDA_SUPPORT
template <typename T>
DataLocation Data<T>::require_gpu() {
/**
Allocate GPU memory in case of GPU array is not inited yet
*/
  if (gpuArr == nullptr) {
    gpuArr.reset(new GNAcuGpuArray<T>());
  }
  if (gpuArr->deviceMemAllocated) {
#ifdef CU_DEBUG_2
    std::cerr << "INITED! Nothing to do! Reqire_gpu exit!" << std::endl;
#endif
    return gpuArr->dataLoc;
  }
  DataLocation tmp = DataLocation::NoData;
  if (type.shape.size() == 1) {
    tmp = gpuArr->Init(type.shape[0], buffer);
  } else if (type.shape.size() == 2) {
    tmp = gpuArr->Init(type.shape[0]*type.shape[1], buffer);
  } 
  return tmp;
}
#endif // GNA_CUDA_SUPPORT
#endif // DATA_H
