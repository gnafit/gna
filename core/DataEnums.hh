#pragma once

namespace GNA {
	enum class MatrixFormat {
		Regular        = 0, ///< Matrix is stored in 2d array
		PermitDiagonal = 1, ///< Dagonal matrix may be defined via 1d array
	};
}
