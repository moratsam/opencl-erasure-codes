# erasure codes with OpenCL in go

## The repo provides vanilla and OpenCL (GPU) implementations of the Reed-Solomon erasure coding over the Galois field GF(2^8).

 
## The README mostly delves into the mathematical background of the project, including:

* Finite fields (refresher + some important lemmas)
* Galois field arithmetic (refresher + examples)
* Reed-Solomon erasure codes (explanation + important proofs).


*Polynomial Codes over Certain Finite Fields*
DOI: 10.1137/0108018

*Optimizing Cauchy Reed-Solomon Codes for Fault-Tolerant Storage Applications* 
DOI: 10.1.1.140.2267

#### Field
set of elements with (+, \*)
with (+, \*) identities
with (+, \*) inverses
division by id(+) not defined.


#### Finite field Zp:
* Lemma 1: Rows in permutation table except row 0 are permutations of [p-1].

PROOF: Suppose not. Suppose x*a = x*b. Then x(a-b) = 0 is divisor of zero. //

#### Galois field
	GF(p^m) are polynomials of degree m-1 over Zp. For example, ax^m-1 + bx^m-2 +..+ f where {a,..f} in [p-1]. 
	Addition and multiplication of the coefficients (but not the polynomials) are defined by Zp.
		addition table for Z2 (XOR)
			+	0	1
			0	0	1
			1	1	0

		multiplication table for Z2 (AND)
			*	0	1
			0	0	0
			1	0	1

	Problem seems to arise: multiplication on polynomials is not closed.

	A **prime** for GF(p^m) is a degree m polynomial that is irreducible over p . This simply means that it cannot be factored. For example, x^3 + 1 is not irreducible over 2 because it can be factored as (x^2 + x + 1)(x + 1).
	If an irreducible polynomial g(x) can be found, then polynomial multiplication can be defined as standard polynomial multiplication modulo g(x).
```
Example for GF(2^3), g(x) = x^3 + x + 1

Dec	Bin	Poly

0		000	0
1		001	1
2		010	x
3		011	x + 1
4		100	x^2
5		101	x^2 + 1
6		110	x^2 + x
7		111	x^2 + x + 1

5*6 != 30 % 8 = 6
5*6 = (x^2 + 1)(x^2 + x) % x^3 + x + 1 = x + 1 = 3
```

#### Galois field arithmetic

GF(2^k) addition or subtraction is xor.
To multiply *a* with *b*, imagine the binary written form as a polynomial of some *x* over {0,1}. Wherever there is a '1' in *a* it means add to the final result that power of *x* multiplied by *b*. Which of course translates to just right shift b by that power. This is done for each '1' in *a*. And how are these partial results then added together? Still thinking of the polynomial representation, it becomes obvious that the simply need to be summed up which is just XOR. Thus, multiplication can be easily implemented with a series of bit shifts and XORs.

The outcome of this operation must by divided by the prime polynomial to ensure that the end result remains in GF(2^k). Now, thinking again in terms of polynomials, division is just subtraction of the divisor at the appropriate powers. And subtraction is also just XOR. The process stops once the the remainder is under 2^k, because for every *e* in GF(2^k): e divided by the prime is *e*.

example: a=33, b=191, prime=0x11d
```
  00100001 #a
* 10111111 #b
  =================
   _____10111111 #this is the rightmost '1' in a; the free coefficient in the polynomial so just *b* multiplied by 1
^ 10111111_____ #this is the second '1'. Here x is raised to the power of 5 so just shift b 5 times. 
  1011101011111 #normal multiplication is finished, result exceeds 2^8 -1


	
  # I try to align the divisor with the first 1 from the dividend, to ease understanding
  1011101011111 #it needs to be divided by the prime to arrive back in GF(2^8)
/ 100011101____ #this is 0x11d
  =================
  0011010001111 #still > 2^8 -1, so repeat
^   100011101__ #I al
  0001011111011 #repeat
^    100011101_
  0000011000001 # = 193 < 2^8 -1, end
```

A generator *g* of a field is an element of the field such that every other element of the field can be expressed as a series of iterative multiplications of *g*. In this way, *g* is said to generate the field.
To optimise multiplication, one can keep in memory the log and exp tables of a generator. Any multiplication in the field can then be performed by two lookups into the log table and 1 lookup into the exp table:

	a*b = g^(logg(a*b)) = g^(logg(a) + logg(b))
	
#### Reed-Solomon
Suppose we have a file of data of size N and suppose we want to create from it (n+k) shards of size roughly N/n, such that possessing any n-subset of the shards allows one to reconstruct the original data. 

note: all operations are performed on Galois fields (in my case GF(2^8) since I'm operating on bytes)

ENCODING: 
i) Create a cool matrix mat of dimensions (n+k)xn.
ii) Divide the data into words of size n (n-words) and stack them into a matrix [data] with dimensions nx(N/n)
iii) Define mat * [data] = [enc] //dimensions of [enc] are obviously [n+k]x[N/n]
Each row in [enc] can be thought of as a shard.
The index of the row should be put into the shard, as it's needed for decoding.


DECODING:
This is the magic idea:
	mat * [data] = [enc] ==> 
	==> mat^-1 * mat * [data] = mat^-1 * [enc] ==>
	==> [data] = mat^-1 * [enc]


Say one has n shards. 
i) stack them together to create a submatrix of [enc] called [subenc]
ii) Create mat
iii) Remove all rows not pertaining to one's shards // now one is left with a nxn submat
iv) Calculate submat^-1, the inverse of submat
v) Reconstruct [data] by multiplying submat^-1 * [subenc]


The main difficulty with these scheme is that mat must have the property that every possible submat must be invertible. I used a standard cauchy matrix for this purpose. Authors recommend appending an identity matrix to the top, to cleanly separate the encoded data into data shards and parity shards. I disregarded this and used a complete cauchy matrix so every shard is encoded.

I implemented the matrix inversion using LU decomposition, of course with the twist that matrix values are polynomials over GF(2^8) and all operations also take place in that field.


![ec1](https://user-images.githubusercontent.com/43090095/138609366-a6258490-2764-4a07-8c83-9b28ee44b800.jpg)

![ec2](https://user-images.githubusercontent.com/43090095/138609362-78657857-64a6-4241-ac90-8cefb1d2dd4f.jpg)


Some notes:

* Any sumbatrix of my original cauchy matrix is a viable cauchy matrix.
* Picture 1 rewrites a cauchy matrix as a product using vandermonde matrixes.
* Picture 2 proves that a cauchy matrix is invertible by providing a formula for the determinant of a (square) cauchy matrix.
* The derivation defines a cauchy matrix as subtracting different indexed items, whereas I implemented it as additions. The derivation still applies because under GF subtraction and addition are equivalent.
* Since x and y terms are pairwise-disjunct by definition, the derived determinant will always be different from zero, which means it is indeed invertible.

