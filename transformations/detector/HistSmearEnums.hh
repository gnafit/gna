#pragma once

namespace GNA {
	enum class DataPropagation {
		Ignore    = 0, ///< Do not propagate the computation result
		Propagate = 1, ///< Propagate the computation result
	};

	enum class DataMutability {
		Static  = 0, ///< data is static, i.e. unchangable
		Dynamic = 1, ///< data is dynamic
	};

	enum class MatrixType {
		Any             = 0b000, ///< Any matrix
		Square          = 0b001, ///< Square matrix
		// LowerTriangular = 0b011, ///< Matrix is lower triangular
		// UpperTriangular = 0b101, ///< Matrix is upper triangular
	};

	enum class SquareMatrixType {
		Any             = 0b00, ///< Any matrix
		LowerTriangular = 0b01, ///< Matrix is lower triangular
		UpperTriangular = 0b10, ///< Matrix is upper triangular
	};

	enum class Spilling {
		None      = 0b00, ///< Do not permit underflows/overflows
		Underflow = 0b01, ///< Permit only underflow
		Overflow  = 0b10, ///< Permit only overflow
		Both      = 0b11, ///< Permit both underflow and overflow
	};
}
