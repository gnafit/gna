#pragma once

#include <cstddef>

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
#include "GpuArray.hh"
#include "DataLocation.hh"
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

  DataType() = default;                                         ///< Constructor.

  /**
   * @brief Copy constructor.
   *
   * Copy constructor copies all members INCLUDING preallocated buffer.
   *
   * @param other instance of DataType.
   */
  DataType(const DataType& other) :
  kind{other.kind}, shape(other.shape), edges(other.edges), edgesNd(other.edgesNd), buffer(other.buffer)
  { }

  /**
   * @brief Copy assignment.
   *
   * Copy assignment copies all members EXCEPT preallocated buffer.
   *
   * @param other instance of DataType.
   */
  DataType& operator=(const DataType& other) {
    kind=other.kind;
    shape=other.shape;
    edges=other.edges;
    edgesNd=other.edgesNd;
    return *this;
  }

  bool operator==(const DataType &other) const;                 ///< Check if data types are identical.
  bool operator!=(const DataType &other) const;                 ///< Check if data types are not identical.
  bool requiresReallocation(const DataType &other) const;       ///< Check if data types are identical.

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
  void preallocated(double *buf) { buffer = static_cast<void*>(buf); }
  void preallocated(float *buf)  { buffer = static_cast<void*>(buf); }
  void preallocated(int *buf)    { buffer = static_cast<void*>(buf); }
  void preallocated(size_t *buf) { buffer = static_cast<void*>(buf); }

  bool point_same(const DataType& other) {
      return this->buffer == other.buffer;
  };

  DataKind kind = DataKind::Undefined;                         ///< DataKind: points (array) or histogram?
  std::vector<size_t> shape;                                   ///< Data dimensions.

  std::vector<double> edges = {};                              ///< Bin edges for 1D histogram.
  std::vector<std::vector<double>> edgesNd = {};               ///< Bin edges for ND histogram. TODO: resolve redundancy of edges and edgesNd
  //std::pair<double, double> bounds = {
    //-std::numeric_limits<double>::infinity(),
     //std::numeric_limits<double>::infinity()
  //};

  void *buffer = nullptr;                                      ///< Preallocated data buffer (double).
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
  T cast()     const { return m_type; }                            ///< Cast to (return) the initialized DataType instance.

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
      fprintf(stderr, ", shape == empty\n");
      break;
    case 1:
      fprintf(stderr, ", shape == (%lu)\n", shape()[0]);
      break;
    case 2:
      fprintf(stderr, ", shape == (%lu, %lu)\n", shape()[0], shape()[1]);
      break;
    default:
      fprintf(stderr, ", shape == WTF(%lu)?\n", shape().size());
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
   * @brief Set the DataType to be the view on the preallocated buffer
   * @param buf -- double buffer
   * @return `*this`.
   */
  DataType::Points<T> &preallocated(double* buf) {
    m_type.preallocated(buf);
    return setKind();
  }

  /**
   * @brief Set the DataType to be the view on the preallocated buffer
   * @param buf -- double buffer
   * @return `*this`.
   */
  DataType::Points<T> &preallocated(float* buf) {
    m_type.preallocated(buf);
    return setKind();
  }

  /**
   * @brief Set the DataType to be the view on the preallocated 1d Array
   * @param array - 1d Eigen array.
   * @return `*this`.
   */
  DataType::Points<T> &view(Eigen::ArrayXd& array) {
    shape(array.size());
    return preallocated(array.data());
  }

  /**
   * @brief Set the DataType to be the view on the preallocated 2d Array
   * @param array - 2d Eigen array.
   * @return `*this`.
   */
  DataType::Points<T> &view(Eigen::ArrayXXd& array) {
    shape(array.rows(), array.cols());
    return preallocated(array.data());
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
  T cast()     const { return m_type; }                            ///< Cast to (return) the initialized DataType instance.

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
    if(m_type.edgesNd.size() != other.m_type.edgesNd.size()){
      return false;
    }
    for (size_t i = 0; i < m_type.edgesNd.size(); ++i)
    {
      if (m_type.edgesNd[i] != other.m_type.edgesNd[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Prints to stderr the dimensions and bin edges of the data.
   */
  void dump() const {
    fprintf(stderr, ", bins X == [%lu]", m_type.shape[0]);
    if(m_type.shape.size()>1){
      fprintf(stderr, ", bins Y == [%lu]", m_type.shape[1]);
    }
    for (size_t i = 0; i < m_type.edgesNd.size(); ++i) {
      auto& edges=m_type.edgesNd[i];
      if(edges.size()>1){
        fprintf(stderr, ", Edges%i[%lu] (%g->%g)", int(i), edges.size(), edges.front(), edges.back());
      }
      else{
        fprintf(stderr, ", Edges%i[%lu]", int(i), edges.size());
      }
    }
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
   * @brief Set the number of bins (2d).
   * @param xbins -- the number of X bins.
   * @param ybins -- the number of Y bins.
   * @return `*this`.
   */
  DataType::Hist<T> &bins(size_t xbins, size_t ybins) {
    m_type.shape = std::vector<size_t>{xbins, ybins};
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
    m_type.edgesNd.resize(1);
    m_type.edgesNd[0]=m_type.edges;
    return bins(edges.size()-1);
  }

  /**
   * @brief Set the bin edges via std::vector (2d).
   * @param xedges -- vector with X bin edges.
   * @param yedges -- vector with Y bin edges.
   * @return `*this`.
   */
  DataType::Hist<T> &edges(const std::vector<double> &xedges, const std::vector<double> &yedges) {
    m_type.edges = xedges;
    m_type.edgesNd.resize(2);
    m_type.edgesNd[0]=xedges;
    m_type.edgesNd[1]=yedges;
    return bins(xedges.size()-1, yedges.size()-1);
  }

  /**
   * @brief Set the bin edges via double buffers (2d).
   * @param nx -- number of X bin edges.
   * @param xedges -- buffer with X bin edges.
   * @param ny -- number of Y bin edges.
   * @param yedges -- buffer with Y bin edges.
   * @return `*this`.
   */
  DataType::Hist<T> &edges(size_t nx, double* xedges, size_t ny, double* yedges) {
    m_type.edges.assign(xedges, xedges+nx);
    m_type.edgesNd.resize(2);
    m_type.edgesNd[0]=m_type.edges;
    m_type.edgesNd[1].assign(yedges, yedges+ny);
    return bins(nx-1, ny-1);
  }

  /**
   * @brief Set the bin edges via double buffer.
   * @param n -- number of bin edges.
   * @param edges -- buffer with bin edges.
   * @return `*this`.
   */
  DataType::Hist<T> &edges(size_t n, double* edges) {
    m_type.edges.assign(edges, edges+n);
    m_type.edgesNd.resize(1);
    m_type.edgesNd[0]=m_type.edges;
    return bins(n-1);
  }

  /**
   * @brief Set the bin edges via double buffer.
   * @param n -- number of bin edges.
   * @param edges -- buffer with bin edges.
   * @return `*this`.
   */
  DataType::Hist<T> &edges(size_t n, const double* edges) {
    m_type.edges.assign(edges, edges+n);
    m_type.edgesNd.resize(1);
    m_type.edgesNd[0]=m_type.edges;
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

  /**
   * @brief Set the DataType to be the view on the preallocated buffer
   * @param buf -- double buffer
   * @return `*this`.
   */
  DataType::Hist<T> &preallocated(double* buf) {
    m_type.preallocated(buf);
    return setKind();
  }

  /**
   * @brief Set the DataType to be the view on the preallocated buffer
   * @param buf -- double buffer
   * @return `*this`.
   */
  DataType::Hist<T> &preallocated(float* buf) {
    m_type.preallocated(buf);
    return setKind();
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
 * for further comparison. Does not check the buffer pointer;
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
 * @brief Check if data requires reallocation.
 *
 * Similar to operator==(), but also checks the buffer pointer.
 */
inline bool DataType::requiresReallocation(const DataType &other) const {
  if( *this != other ){
    return true;
  }

  return buffer!=other.buffer;
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

namespace GNA{
  namespace GNAObjectTemplates{
    template<typename FloatType>
    class ViewRearT;
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
private:
  /// fixme: this is an ugly implementation of viewrear.
  /// the mechanism should be provided on the lavel of framework.
  friend class GNA::GNAObjectTemplates::ViewRearT<T>;
public:
  using ArrayType      = Eigen::Array<T, Eigen::Dynamic, 1>;
  using VectorType     = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using Array2Type     = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixType     = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using ArrayViewType  = Eigen::Map<ArrayType>;
  using VectorViewType = Eigen::Map<VectorType>;
  using Array2ViewType = Eigen::Map<Array2Type>;
  using MatrixViewType = Eigen::Map<MatrixType>;

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
      this->init(static_cast<T*>(dt.buffer));
    }
    else {
      allocated.reset(new T[dt.size()]);
      this->init(allocated.get());
    }
  }
#ifdef GNA_CUDA_SUPPORT
  DataLocation require_gpu();
  DataLocation require_gpu(DataLocation location);
#endif // GNA_CUDA_SUPPORT

  const DataType type;                             ///< data type.
  Status state{Status::Undefined};                 ///< data status.

  T *buffer{nullptr};                              ///< the buffer. Data ownership is undefined.
  std::unique_ptr<T> allocated{nullptr};           ///< the buffer initialized within Data. Deallocates the data when destructed.

  ArrayViewType  arr{nullptr,   0}; ///< 1D array view.
  VectorViewType vec{nullptr,   0}; ///< 1D vector view.

  Array2ViewType arr2d{nullptr, 0,  0}; ///< 2D array  view.
  MatrixViewType mat{nullptr,   0,  0}; ///< 2D matrix view.

  ArrayViewType  &x = arr;    ///< 1D array  view shorthand.
#ifdef GNA_CUDA_SUPPORT
  std::unique_ptr<GpuArray<T>> gpuArr{nullptr};    ///< container for data on GPU, view to GPU array
#endif


protected:
  void init(T* buffer){
    this->buffer = buffer;
    if (type.shape.size() == 1) {
      new (&this->arr)   ArrayViewType(  this->buffer, type.shape[0] );
      new (&this->vec)   VectorViewType( this->buffer, type.shape[0] );
    } else if (type.shape.size() == 2) {
      new (&this->arr)   ArrayViewType(  this->buffer, type.shape[0]*type.shape[1] );
      new (&this->vec)   VectorViewType( this->buffer, type.shape[0]*type.shape[1] );

      new (&this->arr2d) Array2ViewType( this->buffer, type.shape[0], type.shape[1] );
      new (&this->mat)   MatrixViewType( this->buffer, type.shape[0], type.shape[1] );
    }
  }
};

#ifdef GNA_CUDA_SUPPORT

/**
Allocate GPU memory in case of GPU array is not inited yet
*/
template <typename T>
DataLocation Data<T>::require_gpu() {
  if (gpuArr == nullptr) {
    gpuArr.reset(new GpuArray<T>());
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

template <typename T>
DataLocation Data<T>::require_gpu(DataLocation location) {
  require_gpu();
  gpuArr->setLocation(location);
  return location;
}
#endif

template class Data<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class Data<float>;
#endif
