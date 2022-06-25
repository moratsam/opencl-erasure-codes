package pu

type PU interface {
	// inv: n x n, enc: n x d, returns d n-words of decoded data.
	Decode(inv, enc [][]byte) ([]byte, error)

	// cauchy: n x n+k, data: n*n-word (n n-words).
	// Returns n+k x n-word (n+k shards of n-word encoded bytes).
	Encode(cauchy [][]byte, data []byte) ([][]byte, error)
}

