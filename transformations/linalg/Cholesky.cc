#include "Cholesky.hh"

using GNA::MatrixFormat;

Cholesky::Cholesky(MatrixFormat matrix_format) :
m_permit_diagonal{matrix_format==MatrixFormat::PermitDiagonal}
{
	transformation_("cholesky")
		.input("mat")
		.output("L")
		.types(&Cholesky::prepareCholesky)
		.func(&Cholesky::calculateCholesky)
		;
}

/**
 * Check that the input is matrix and the matrix is symmetric
 */
void Cholesky::prepareCholesky(TypesFunctionArgs& fargs) {
    auto& arg = fargs.args[0];
    auto& ret = fargs.rets[0];
    ret=arg;
    switch (arg.shape.size()){
        case 2:
            if (arg.shape[0]!=arg.shape[1]){
				throw fargs.args.error(arg, "Cholesky input matrix should square");
            }
            m_llt.reset(new LLT(arg.shape[0]));
            ret.preallocated(const_cast<double*>(m_llt->matrixRef().data()));
            break;
        case 1:
			if(!m_permit_diagonal){
				throw fargs.args.error(arg, "Cholesky input should be a matrix. Enable `permit_diagonal` to work with 1d inputs");
			}
            break;
        default:
            throw fargs.args.error(arg);
    }
}

/**
 * Decompose
 */
void Cholesky::calculateCholesky(FunctionArgs& fargs) {
    auto& arg = fargs.args[0];
    switch (arg.type.shape.size()){
        case 2:
            m_llt->compute(arg.mat);
            break;
        case 1:
            fargs.rets[0].x = arg.x.sqrt();
            break;
        default: // Should not happen as it is checked in prepareCholesky
            throw fargs.rets.error("Cholesky: invalid input dimension");
    }
}
