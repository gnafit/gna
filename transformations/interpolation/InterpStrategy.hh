#pragma once

namespace GNA {
    namespace Interpolation {
          enum class Strategy { ///< Extrapolation strategy.
            Constant = 0,       ///< Fill with constant value.
            Extrapolate = 1     ///< Extrapolate using first/last segment function.
          };
    }
}
