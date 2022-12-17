"""
A component that simulates round 3 of the PLONK proving system.
"""

from typing import List
import time
import random

# from py_ecc.bls import G2ProofOfPossession as bls12_381_G2ProofOfPossession




# Represent the 8 polynomial interpolations as vectors of coefficients
# for a n circuits, a,b,c are of length n+2 and the rest are of length n for this simulation
class Round3:
    def __init__(self,N):
        self.N = N
        self.a = [1]*(self.N + 2)
        self.b = [1]*(self.N + 2)
        self.c = [1]*(self.N + 2)
        self.qm = [1]*self.N
        self.ql = [1]*self.N
        self.qr = [1]*self.N
        self.qo = [1]*self.N
        self.qc = [1]*self.N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k1 = 1
        self.k2 = 1
        self.z = [1]*(self.N+3)
        self.s1 = [1]*self.N
        self.s2 = [1]*self.N
        self.s3 = [1]*self.N
        self.l1 = [1]*self.N

        self.setAllVectorsToRandomEntries()
        self.setAllConstantsToRandomValues()

    def setAllConstantsToRandomValues(self):
        # Set all constants to random values
        self.alpha = random.randint(0, 100)
        self.beta = random.randint(0, 100)
        self.gamma = random.randint(0, 100)
        self.k1 = random.randint(0, 100)
        self.k2 = random.randint(0, 100)


    def setAllVectorsToRandomEntries(self):
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
        result = [0] * (len(poly1) + len(poly2) - 1)
        for i in range(len(poly1)):
            for j in range(len(poly2)):
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
        subterm2 = self.multiplyManyPolynomials([self.a,self.ql])
        subterm3 = self.multiplyManyPolynomials([self.b,self.qr])
        subterm4 = self.multiplyManyPolynomials([self.c, self.qo])

        return subterm1 + subterm2 + subterm3 + subterm4

    def getTerm2(self):
        subterm1 =  self.addManyPolynomials([self.alpha*self.a, self.alpha*self.beta* [1,1],self.alpha*self.gamma ])
        subterm2 =  self.addManyPolynomials([self.alpha*self.b, self.alpha*self.beta*self.k1*[1,1],self.alpha*self.gamma ])
        subterm3 =  self.addManyPolynomials([self.alpha*self.c, self.alpha*self.beta*self.k2*[1,1],self.alpha*self.gamma ])

        return self.multiplyManyPolynomials([subterm1, subterm2, subterm3, self.z])

    def getTerm3(self):
        subterm1 =  self.addManyPolynomials([self.alpha*self.a, self.alpha*self.beta*self.s1,self.alpha*self.gamma ])
        subterm2 =  self.addManyPolynomials([self.alpha*self.b, self.alpha*self.beta*self.s2,self.alpha*self.gamma ])
        subterm3 =  self.addManyPolynomials([self.alpha*self.c, self.alpha*self.beta*self.s3,self.alpha*self.gamma ])

        # This in practice should be z(xw), but its not needed for the simulation
        return self.multiplyManyPolynomials([subterm1, subterm2, subterm3, self.z]) 

    def getTerm4(self):
        subterm = self.addPolynomials([self.alpha**2*self.z, -self.alpha**2])
        return self.multiplyManyPolynomials([subterm, self.l1])

    def timeRound3(self):
        # Measure the time it takes to run the serial implementation
        start = time.time()
        self.serialImplementation()
        end = time.time()
        return end - start
    


    

