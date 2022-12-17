"""
A component that simulates round 3 of the PLONK proving system.
"""

from typing import List
import time
import random

# from py_ecc.bls import G2ProofOfPossession as bls12_381_G2ProofOfPossession


class Helpers:
    def __init__(self):
        ...

    def genAllConstantValues(self) -> dict:
        ub = 100
        # Generate all constant values
        alpha = random.randint(0, ub)
        beta = random.randint(0, ub)
        gamma = random.randint(0, ub)
        k1 = random.randint(0, ub)
        k2 = random.randint(0, ub)
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "k1": k1, "k2": k2}

    def genAllVectorValues(self, N) -> dict:
        ub = 100
        # Generate all vector values
        a = [random.randint(0, ub) for i in range(N+2)]
        b = [random.randint(0, ub) for i in range(N+2)]
        c = [random.randint(0, ub) for i in range(N+2)]
        qm = [random.randint(0, ub) for i in range(N)]
        ql = [random.randint(0, ub) for i in range(N)]
        qr = [random.randint(0, ub) for i in range(N)]
        qo = [random.randint(0, ub) for i in range(N)]
        qc = [random.randint(0, ub) for i in range(N)]
        s1 = [random.randint(0, ub) for i in range(N)]
        s2 = [random.randint(0, ub) for i in range(N)]
        s3 = [random.randint(0, ub) for i in range(N)]
        z = [random.randint(0, ub) for i in range(N+3)]
        l1 = [random.randint(0, ub) for i in range(N)]
        return {"a": a, "b": b, "c": c, "qm": qm, "ql": ql, "qr": qr, "qo": qo, "qc": qc, "s1": s1, "s2": s2, "s3": s3, "z": z, "l1": l1}


# Represent the 8 polynomial interpolations as vectors of coefficients
# for a n circuits, a,b,c are of length n+2 and the rest are of length n for this simulation
class Round3:
    def __init__(self, N, vectors=None, constants=None):
        self.N = N
        self.a = [1]*(self.N + 2)
        self.b = [1]*(self.N + 2)
        self.c = [1]*(self.N + 2)
        self.qm = [1]*self.N
        self.ql = [1]*self.N
        self.qr = [1]*self.N
        self.qo = [1]*self.N
        self.qc = [1]*self.N
        self.k1 = 1
        self.k2 = 1
        self.z = [1]*(self.N+3)
        self.s1 = [1]*self.N
        self.s2 = [1]*self.N
        self.s3 = [1]*self.N
        self.l1 = [1]*self.N

        if vectors is None:
            self.setAllVectorsToRandomEntries()
        else:
            self.setAllVectors(vectors)
        if constants is None:
            self.setAllConstantsToRandomValues()
        else:
            self.setAllConstants(constants)

    def setAllConstantsToRandomValues(self) -> dict:
        # Set all constants to random values
        self.alpha = random.randint(0, 100)
        self.beta = random.randint(0, 100)
        self.gamma = random.randint(0, 100)
        self.k1 = random.randint(0, 100)
        self.k2 = random.randint(0, 100)
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "k1": self.k1, "k2": self.k2}

    def setAllVectorsToRandomEntries(self) -> dict:
        # Set all vectors to random entries
        self.a = [random.randint(0, 100) for i in range(len(self.a))]
        self.b = [random.randint(0, 100) for i in range(len(self.b))]
        self.c = [random.randint(0, 100) for i in range(len(self.c))]
        self.qm = [random.randint(0, 100) for i in range(len(self.qm))]
        self.ql = [random.randint(0, 100) for i in range(len(self.ql))]
        self.qr = [random.randint(0, 100) for i in range(len(self.qr))]
        self.qo = [random.randint(0, 100) for i in range(len(self.qo))]
        self.qc = [random.randint(0, 100) for i in range(len(self.qc))]
        self.s1 = [random.randint(0, 100) for i in range(len(self.s1))]
        self.s2 = [random.randint(0, 100) for i in range(len(self.s2))]
        self.s3 = [random.randint(0, 100) for i in range(len(self.s3))]
        self.z = [random.randint(0, 100) for i in range(len(self.z))]
        self.l1 = [random.randint(0, 100) for i in range(len(self.l1))]
        return {"a": self.a, "b": self.b, "c": self.c, "qm": self.qm, "ql": self.ql, "qr": self.qr, "qo": self.qo, "qc": self.qc, "s1": self.s1, "s2": self.s2, "s3": self.s3, "z": self.z, "l1": self.l1}

    def setAllConstants(self, constant_dict):
        # Set all constants to values in dict
        self.alpha = constant_dict["alpha"]
        self.beta = constant_dict["beta"]
        self.gamma = constant_dict["gamma"]
        self.k1 = constant_dict["k1"]
        self.k2 = constant_dict["k2"]

    def setAllVectors(self, vector_dict):
        # Set all vectors to values in dict
        self.a = vector_dict["a"]
        self.b = vector_dict["b"]
        self.c = vector_dict["c"]
        self.qm = vector_dict["qm"]
        self.ql = vector_dict["ql"]
        self.qr = vector_dict["qr"]
        self.qo = vector_dict["qo"]
        self.qc = vector_dict["qc"]
        self.s1 = vector_dict["s1"]
        self.s2 = vector_dict["s2"]
        self.s3 = vector_dict["s3"]
        self.z = vector_dict["z"]
        self.l1 = vector_dict["l1"]

    def serialImplementation(self):
        # Calculate 4 terms and add them together
        term1 = self.getTerm1()
        term2 = self.getTerm2()
        term3 = self.getTerm3()
        term4 = self.getTerm4()
        return term1 + term2 - term3 + term4

    def multiplyPolynomials(self, poly1, poly2):
        # Multiply two polynomials
        # poly1 and poly2 are lists of coefficients
        # The result is a list of coefficients
        print(f"{poly1}\n{poly2}")
        time.sleep(.5)
        result = [0] * (len(poly1) + len(poly2) - 1)
        for i in range(len(poly1)):
            for j in range(len(poly2)):
                print(f"{i}\n{j}")
                result[i + j] += poly1[i] * poly2[j]
        return result

    def multiplyManyPolynomials(self, polys):
        # Multiply many polynomials
        # polys is a list of polynomials, each polynomial is a list of coefficients
        # The result is a list of coefficients
        result = [0]
        for poly in polys:
            result = self.multiplyPolynomials(result, poly)
        return result

    def addPolynomials(self, poly1, poly2):
        # Add two polynomials
        # poly1 and poly2 are lists of coefficients
        # The result is a list of coefficients
        result = [0] * max(len(poly1), len(poly2))
        for i in range(len(poly1)):
            result[i] += poly1[i]
        for i in range(len(poly2)):
            result[i] += poly2[i]
        return result

    def addManyPolynomials(self, polys):
        # Add many polynomials
        # polys is a list of polynomials, each polynomial is a list of coefficients
        # The result is a list of coefficients
        result = [0]
        for poly in polys:
            result = self.addPolynomials(result, poly)
        return result

    def getTerm1(self):
        subterm1 = self.multiplyManyPolynomials([self.a, self.b, self.qm])
        subterm2 = self.multiplyManyPolynomials([self.a, self.ql])
        subterm3 = self.multiplyManyPolynomials([self.b, self.qr])
        subterm4 = self.multiplyManyPolynomials([self.c, self.qo])

        return subterm1 + subterm2 + subterm3 + subterm4

    def getTerm2(self):
        subterm1 = self.addManyPolynomials(
            [self.alpha*self.a, self.alpha*self.beta * [1, 1], [self.alpha*self.gamma]])
        subterm2 = self.addManyPolynomials(
            [self.alpha*self.b, self.alpha*self.beta*self.k1*[1, 1], [self.alpha*self.gamma]])
        subterm3 = self.addManyPolynomials(
            [self.alpha*self.c, self.alpha*self.beta*self.k2*[1, 1], [self.alpha*self.gamma]])

        return self.multiplyManyPolynomials([subterm1, subterm2, subterm3, self.z])

    def getTerm3(self):
        subterm1 = self.addManyPolynomials(
            [self.alpha*self.a, self.alpha*self.beta*self.s1, [self.alpha*self.gamma]])
        subterm2 = self.addManyPolynomials(
            [self.alpha*self.b, self.alpha*self.beta*self.s2, [self.alpha*self.gamma]])
        subterm3 = self.addManyPolynomials(
            [self.alpha*self.c, self.alpha*self.beta*self.s3, [self.alpha*self.gamma]])

        # This in practice should be z(xw), but its not needed for the simulation
        return self.multiplyManyPolynomials([subterm1, subterm2, subterm3, self.z])

    def getTerm4(self):
        subterm = self.addPolynomials([self.alpha**2*self.z, -self.alpha**2])
        return self.multiplyManyPolynomials([subterm, self.l1])

    def runRound3(self):
        # Measure the time it takes to run the serial implementation
        start = time.time()
        out = self.serialImplementation()
        end = time.time()
        return out, end - start
