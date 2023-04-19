from load import ROOT as R
import pytest

def test_errors_01():
	with pytest.raises(R.KeyError):
		R.GNAUnitTest.test_keyerror_exception();

	with pytest.raises(R.TransformationTypes.TypeError):
		R.GNAUnitTest.test_typeerror_exception();

def test_errors_02():
	trans = R.DebugTransformation('test exception')
	trans.emit_calculation_error=True

	with pytest.raises(R.std.runtime_error):
		trans.single().data()
