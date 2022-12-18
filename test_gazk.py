import pytest
from gazk.components.round3 import Helpers, Round3
from gazk.components.round3cuda import ParallelRound3
from gazk.components.round3fft import Round3FFT



def test_add_two_polynomials():
    a = [1,3,4]
    b = [1,5,3]
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.addPolynomials(a,b)
    print(result)

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
    print(result)

def test_multiply_polynomials():
    a = [1,3,4]
    b = [1,5,3]
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    handler = Round3(N, vectors=vectors, constants=constants)

    result = handler.multiplyPolynomials(a,b)
    print(result)

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
    print(result)

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
    p = ParallelRound3()
    handler = p.naive()

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


def test_naive_implementation():
    # run naive implementation
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    p = ParallelRound3()
    res, time = p.naive(vectors['a'], vectors['b'])
    print(res)
    print("Time taken: ", time)

def test_mapreduce_implementation():
    # run mapreduce implementation
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    # print(f"{constants=}{vectors=}")
    p = ParallelRound3()
    res, time = p.polymult_with_scan(vectors['a'], vectors['b'])
    print(res)
    print("Time taken: ", time)

def test_pymultfft_implementation():
    # run pymultfft implementation
    N = 2
    helpers = Helpers()
    constants = helpers.genAllConstantValues()
    vectors = helpers.genAllVectorValues(N)
    p = Round3FFT()
    res, time = p.fft(N)
    print(res)
    print("Time taken: ", time)


if __name__=="__main__":
    test_serial_implementation()
    # test_getTerm1()
