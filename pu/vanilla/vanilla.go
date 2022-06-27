package vanilla

import (
	"sync"
	
	u "github.com/moratsam/opencl-erasure-codes/util"
)

type VanillaPU struct {
}

func NewVanillaPU() *VanillaPU {
	return &VanillaPU{}
}

func (v *VanillaPU) Encode(cauchy [][]byte, data []byte) ([][]byte, error) {
	n := len(cauchy[0])		
	k := len(cauchy)-n
	n_words := len(data)/n

	enc := make([][]byte, n+k)
	for i := range enc {
		enc[i] = make([]byte, n_words)
	}

	// Create function for encoding a single shard.
	wg := new(sync.WaitGroup)
	encodeShard := func(shard_ix int){
		cauchy_row := cauchy[shard_ix]
		for word_ix:=0; word_ix<n_words; word_ix++ { // For every n-word of data
			for ix := range cauchy_row {	// Do dot product of the n-word with the cauchy row.
				enc[shard_ix][word_ix] = u.Add(enc[shard_ix][word_ix], u.Mul(cauchy_row[ix], data[ix+n*word_ix]))
			}
		}
		wg.Done()
	}

	// Spawn n+k routines to encode shards.
	wg.Add(n+k)
	for i := range enc {
		go encodeShard(i)
	}
	wg.Wait()

	return enc, nil
}

func (v *VanillaPU) Decode(inv, enc [][]byte) ([]byte, error){
	n := len(enc)
	n_words := len(enc[0])
	data := make([]byte, 0, n*n_words)

	for word_ix:=0; word_ix<n_words; word_ix++ { // For every n-word of encrypted data
		enc_word := make([]byte, n) // Take one byte from each shard to get an encrypted word.
		for i:=0; i<n; i++ {
			enc_word[i] = enc[i][word_ix]
		}
		data_word := decodeWord(inv, enc_word)
		data = append(data, data_word...)
	}

	return data, nil
}

func decodeWord(inv [][]byte, enc []byte) []byte{
	dim := len(inv[0])

	//calculate W := (L^-1)[enc]
	w := make([]byte, dim)
	for r:=0; r<dim; r++ { //for every row in inv
		for j:=0; j<=r; j++ {
			if r == j { //diagonal values were overwritten in LU, but pretend they're still 1
				w[r] = u.Add(w[r], enc[j])
			} else {
				w[r] = u.Add(w[r], u.Mul(inv[r][j], enc[j]))
			}
		}
	}

	data_word := make([]byte, dim)
	for r:=dim-1; r>=0; r-- {
		for j:=dim-1; j>=r; j-- {
			data_word[r] = u.Add(data_word[r], u.Mul(inv[r][j], w[j]))
		}
	}
	return data_word
}

