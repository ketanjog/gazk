import pytest
from gazk.components.round3 import Helpers, Round3



def test_add_two_polynomials():
    a = [1,3,4]
    b = [1,5,3]
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.addPolynomials(a,b)
    assert result == [2,8,7]

def test_add_many_polynomials():
    a = [1,3,4]
    b = [1,5,3]
    c = [1,2,3]
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.addManyPolynomials([a,b,c])
    assert result == [3,10,10]

def test_multiply_polynomials():
    a = [1,3,4]
    b = [1,5,3]
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.multiplyPolynomials(a,b)
    assert result == [1,8,22,29,12]

def test_multiply_many_polynomials():
    a = [1,3,4]
    b = [1,5,3]
    c = [1,2]
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.multiplyManyPolynomials([a,b,c])
    assert result == [1,10,38,73,70,24]

def test_getTerm1():
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.getTerm1()
    #assert result == [1, 10, 38, 73, 70, 24]
    print(result)

def test_getTerm2():
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.getTerm2()
    #assert result == [1, 10, 38, 73, 70, 24]
    print(result)

def test_getTerm3():
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.getTerm3()
    #assert result == [1, 10, 38, 73, 70, 24]
    print(result)

def test_getTerm4():
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.getTerm4()
    #assert result == [1, 10, 38, 73, 70, 24]
    print(result)

def test_serial_implementation():
	# run serial implementation
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    # print(f"{constants=}{vectors=}")
    round3 = Round3(N, vectors=vectors, constants=constants)
    res, time = round3.runRound3()
    print(res)
    print("Time taken: ", time)



   



if __name__=="__main__":
    test_serial_implementation()
    # test_getTerm1()
