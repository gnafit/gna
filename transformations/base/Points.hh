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
class Points: public GNASingleObject,
              public TransformationBind<Points> {
public:
  /**
   * @brief Construct 1d array from a vector of doubles.
   *
   * The size is determined by the size of the vector.
   *
   * @param points - vector of doubles.
   */
  Points(const std::vector<double> &points) : Points(&points[0], points.size()) {  }

  /**
   * @brief Construct 1d array from C++ array of doubles.
   * @param points - pointer to an array of doubles.
   * @param cnt - the array size.
   */
  Points(const double *points, size_t cnt)
    : m_points(Eigen::Map<const Eigen::ArrayXd>(points, cnt)), m_shape{cnt}
  {
    init();
  }

  /**
   * @brief Construct Nd array from C++ array of doubles.
   * @param points - pointer to an array of doubles.
   * @param shape - vector with dimensions of an array.
   */
  Points(const double *points, std::vector<size_t> shape)
    : m_shape(shape)
  {
    size_t cnt = 1;
    for (auto x: shape) {
      cnt *= x;
    }
    m_points=Eigen::Map<const Eigen::ArrayXd>(points, cnt);

    init();
  }

  /**
   * @brief Construct from just a single point.
   * @param single_point - just one double.
   */
  Points(const double single_point) : Points(&single_point, 1) {}

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
  const double *data() const {
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
    transformation_("points")                           /// Initialize the transformation points.
      .output("points")                                       /// Add an output points.
      .types([](Points *obj, Atypes /*args*/, Rtypes rets) {  /// Define the TypesFunction:
          rets[0] = DataType().points().shape(obj->m_shape);  ///   - assign the data shape for the first output (points).
          rets[0].preallocated(obj->m_points.data());         ///   - tell the DataType that the buffer is preallocated (m_points).
        })
      .func([](Args,Rets){})                                  /// Assign empty Function.
      .finalize();                                            /// Tell the initializer that there are no more configuration and it may initialize the types.
  }
  Eigen::ArrayXd m_points;                                    ///< The array holding the raw 1d data buffer.
  std::vector<size_t> m_shape;                                ///< Vector with data dimensions.
};
