#include "LinearInterpolator.hh"

void LinearInterpolator::indexBins() {
  if (m_xs.size() < 2) {
    return;
  }
  double minbinsize = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < m_xs.size()-1; ++i) {
    minbinsize = std::min(minbinsize, m_xs[i+1] - m_xs[i]);
  }
  m_index.clear();
  m_index.reserve((m_xs.back() - m_xs.front())/minbinsize);
  double x = m_xs.front();
  size_t i = 0;
  for (size_t k = 0; ; ++k) {
    x = m_xs.front() + k*minbinsize;
    if (x > m_xs[i]) {
      if (i+1 == m_xs.size()) {
        m_index.push_back(i);
        break;
      } else {
        i += 1;
      }
    }
    m_index.push_back(i);
  }
  m_minbinsize = minbinsize;
}

void LinearInterpolator::interpolate(FunctionArgs& fargs) {
  const auto &xs = fargs.args[0].x;
  auto &ys = fargs.rets[0].x;
  Eigen::ArrayXi idxes = ((xs - m_xs[0]) / m_minbinsize).cast<int>();
  for (int i = 0; i < idxes.size(); ++i) {
    if (idxes(i) < 0 || static_cast<size_t>(idxes(i)) >= m_index.size()) {
      if (m_status_on_fail == ReturnOnFail::UseZero) {
          ys(i) = 0.;
      } else {
        ys(i) = std::numeric_limits<double>::quiet_NaN();
      }
      continue;
    }
    size_t j;
    for (j = m_index[idxes(i)]; j < m_xs.size(); ++j) {
      if (m_xs[j] <= xs(i) && xs(i) <= m_xs[j+1]) {
        break;
      }
    }
    if (j < m_xs.size()) {
      size_t j2;
      double off = xs(i) - m_xs[j];
      if (off == 0) {
        ys(i) = m_ys[j];
        continue;
      } else if ((off > 0 && j+1 < m_xs.size()) || j == 0) {
        j2 = j+1;
      } else {
        j2 = j-1;
      }
      ys(i) = m_ys[j] + (m_ys[j2]-m_ys[j])/(m_xs[j2]-m_xs[j])*off;
    } else if (m_status_on_fail == ReturnOnFail::UseNaN) {
      ys(i) = std::numeric_limits<double>::quiet_NaN();
    }
    else if (m_status_on_fail == ReturnOnFail::UseZero) {
        ys(i) = 0.;
    }
  }
}
