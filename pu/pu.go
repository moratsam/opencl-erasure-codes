package pu

type PU interface {
	// inv: n x n, enc: n x q, returns q n-words of decoded data.
	Decode(inv, enc [][]byte) ([]byte, error)

	// cauchy: n x n+k, data: q*n-word (q n-words).
	// Returns n+k x q (n+k shards of q encoded bytes).
	Encode(cauchy [][]byte, data []byte) ([][]byte, error)
}

type StreamerPU interface {
	InitDecoder(mat [][]byte) (chan []byte, error)
	Decode([][]byte)
	Encode([]byte)
}

