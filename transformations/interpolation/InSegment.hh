#pragma once

#include "GNAObject.hh"

/**
 * @brief Determines the bins edges indices for an unsorted array.
 *
 * For a given array of bin edges:
 * ```python
 *   edges=(e1, e2, ..., eN)
 * ```
 *
 * And for a given array of points:
 * ```python
 *   points=(p1, p2, ..., pM)
 * ```
 *
 * Determines the indices:
 * ```python
 *   insegment=(n1, n2, ..., nM)
 * ```
 *
 * So that:
 * ```python
 *   e[insegment[n]]<=points[n]<e[insegment[n+1]]
 * ```
 *
 * The transformation is needed in order to implement interpolation,
 * for example InterpExpo.
 *
 * Inputs:
 *   - segments.points - array (to interpolate on).
 *   - segments.edges - edges.
 *
 * Outputs:
 *   - segments.insegment - segment indices.
 *   - segments.widths - segment widths.
 *
 * @note If `edge-point<m_tolerance` is equivalent to `point==edge`.
 *
 * @author Maxim Gonchar
 * @date 02.2018
 */
class InSegment: public GNAObject,
                 public TransformationBind<InSegment> {
public:
  InSegment();                                               ///< Constructor.
  void setTolerance(double value) { m_tolerance = value; }   ///< Set tolerance.
  void determineSegments(FunctionArgs& fargs);               ///< Function that determines segment belonging.

private:
  double m_tolerance{1.e-16};                                ///< Tolerance. If point is below left edge on less than m_tolerance, it is considered to belong to the bin anyway.
};
