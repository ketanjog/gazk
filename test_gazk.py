import pytest
from gazk.components.round3 import Helpers, Round3

def test_serial_implementation():
	# run serial implementation
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    print(f"{constants=}{vectors=}")
    round3 = Round3(N, vectors=vectors, constants=constants)
    res, time = round3.runRound3()
    print(f"{res=}{time=}")


if __name__=="__main__":
    test_serial_implementation()