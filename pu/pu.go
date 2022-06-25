package pu

type PU interface {
	Decode(inv [][]byte, enc []byte) ([]byte, error)
	Encode(cauchy [][], dec []byte) ([]byte, error)
}

