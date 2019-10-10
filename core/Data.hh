#pragma once

#include <cstddef>

#include <functional>
#include <algorithm>
#include <memory>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "config_vars.h"

#ifdef GNA_CUDA_SUPPORT
#include "GpuArray.hh"
#include "DataLocation.hh"
#include "cuda_config_vars.h"
#endif // GNA_CUDA_SUPPORT

#include "DataType.hh"


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
  using ArrayType      = Eigen::Array<T, Eigen::Dynamic, 1> ;
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
