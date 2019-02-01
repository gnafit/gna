#pragma once

#include <vector>
#include "GNAObject.hh"

/**
 * @brief Transformation object holding a static 1- or 2-dimensional array.
 *
 * Outputs:
 *   - `points.points` - 1- or 2- dimensional array with fixed data.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
template<typename FloatType>
class PointsT: public GNASingleObjectT<FloatType,FloatType>,
               public TransformationBind<PointsT<FloatType>,FloatType,FloatType> {
private:
  using BaseClass = GNAObjectT<FloatType,FloatType>;
public:
  using typename BaseClass::FunctionArgs;
  using typename BaseClass::TypesFunctionArgs;
  using ArrayType = Eigen::Array<FloatType,Eigen::Dynamic,1>;
  using PointsType = PointsT<FloatType>;

  /**
   * @brief Construct 1d array from a vector of floats.
   *
   * The size is determined by the size of the vector.
   *
   * @param points - vector of floats.
   */
  PointsT(const std::vector<FloatType> &points) : PointsType(&points[0], points.size()) {  }

  /**
   * @brief Construct 1d array from C++ array of floats.
   * @param points - pointer to an array of floats.
   * @param cnt - the array size.
   */
  PointsT(const FloatType *points, size_t cnt)
    : m_points(Eigen::Map<const ArrayType>(points, cnt)), m_shape{cnt}
  {
    init();
  }

  /**
   * @brief Construct Nd array from C++ array of floats.
   * @param points - pointer to an array of floats.
   * @param shape - vector with dimensions of an array.
   */
  PointsT(const FloatType *points, std::vector<size_t> shape)
    : m_shape(shape)
  {
    size_t cnt = 1;
    for (auto x: shape) {
      cnt *= x;
    }
    m_points=Eigen::Map<const ArrayType>(points, cnt);

    init();
  }

  /**
   * @brief Construct from just a single point.
   * @param single_point - just one FloatType.
   */
  PointsT(const FloatType single_point) : PointsType(&single_point, 1) {}

  /**
   * @brief Return the size of an array.
   * @return number of bins in the histogram.
   */

  size_t size() const {
    return m_points.size();
  }

  /**
   * @brief Return the pointer to C++ array.
   * @return array pointer.
   */
  const FloatType *data() const {
    return m_points.data();
  }

protected:
  /**
   * @brief Initialize the transformation.
   *
   * Transformation `points` with single output `points`.
   *
   * TypeFunction passes the Data shape and the pointer to the buffer with data.
   *
   * The transformation function is empty.
   */
  void init() {
    this->transformation_("points")                                      /// Initialize the transformation points.
      .output("points")                                            /// Add an output points.
      .types([](PointsType *obj, TypesFunctionArgs& fargs) {       /// Define the TypesFunction:
          fargs.rets[0] = DataType().points().shape(obj->m_shape); ///   - assign the data shape for the first output (points).
          fargs.rets[0].preallocated(obj->m_points.data());        ///   - tell the DataType that the buffer is preallocated (m_points).
        })
      .func([](FunctionArgs& fargs){})                             /// Assign empty Function.
      .finalize();                                                 /// Tell the initializer that there are no more configuration and it may initialize the types.
  }
  ArrayType m_points;                                              ///< The array holding the raw 1d data buffer.
  std::vector<size_t> m_shape;                                     ///< Vector with data dimensions.
};

using Points = PointsT<double>;
