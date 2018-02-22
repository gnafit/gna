#pragma once

#include "GNAObject.hh"

/**
 * @brief Determines the bins edges indices for a sorted array.
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
 *   segments=(n1, n2, ..., nN)
 * ```
 *
 * So that:
 * ```python
 *   e1<=points[n1:n2]<e2
 *   e2<=points[n2:n3]<e3
 *   ...
 *   eN-1<=points[nN-1:nN]<eN
 * ```
 *
 * Segments with no points have coinciding indices nI=nJ.
 *
 * The transformation is needed in order to implement interpolation,
 * for example InterpExpo.
 *
 * Inputs:
 *   - segments.points
 *   - segments.edges
 *
 * Outputs:
 *   - segments.segments
 *
 * @note If `edge-point<m_tolerance` is equivalent to `point==edge`.
 *
 * @author Maxim Gonchar
 * @date 02.2018
 */
class SegmentWise: public GNAObject,
                   public TransformationBind<SegmentWise> {
public:
  SegmentWise();                                             ///< Constructor.
  void setTolerance(double value) { m_tolerance = value; }   ///< Set tolerance.
  void determineSegments(Args, Rets);                        ///< Function that determines segments.

private:
  double m_tolerance{1.e-16};                                ///< Tolerance. If point is below left edge on less than m_tolerance, it is considered to belong to the bin anyway.
};
