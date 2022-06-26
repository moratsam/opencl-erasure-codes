package codec

import (
	_"fmt"
	"os"
	"strconv"

	"github.com/moratsam/opencl-erasure-codes/io"
	proc_unit "github.com/moratsam/opencl-erasure-codes/pu"
	u "github.com/moratsam/opencl-erasure-codes/utils"
)

const CHUNK_SIZE = 32*1000


type Codec struct{
	pu	proc_unit.PU
}
func NewCodec(pu proc_unit.PU) *Codec {
	return &Codec{pu}
}

func (c *Codec) Encode(k, n byte, filepath string) error {
	chunk_size := int64(n)*CHUNK_SIZE
	// Open input file.
	f, err := io.OpenFile(filepath)
	if err != nil {
		return err
	}
	defer f.Close()

	// If size of input file modulo n is not 0, create some padding.
	fsize, err :=  io.FileSize(filepath)
	if err != nil {
		return err
	}
	padding := byte((int64(n) - (fsize % int64(n))) % int64(n))

	// Create Cauchy matrix.
	mat := u.CreateCauchy(k, n)

	// Create shards.
	shards := make([]*os.File, int(k+n))
	for i := range shards {
		shards[i], err = io.CreateFile(filepath + "_" + strconv.Itoa(i) + ".enc")	
		if err != nil {
			return err
		}
		defer shards[i].Close()
	}

	// Write metadata to shards.
	if err := metaToShards(shards, n, padding, mat); err != nil {
		return err
	}

	for cnt:=0;;cnt++{
		// Read chunk of input file.
		var chunk []byte
		if cnt == 0 { // First chunk may need to be padded. 
			chunk, err = io.ReadFrom(f, chunk_size-int64(padding))
			chunk = append(make([]byte, int(padding)), chunk...)
		} else {
			chunk, err = io.ReadFrom(f, chunk_size)
		}
		if err != nil {
			return err
		}
		if len(chunk) == 0 { // EOF
			break
		}

		// Encode chunk.
		enc, err := c.pu.Encode(mat, chunk)
		if err != nil {
			return err
		}

		// Write it to shards.
		if err := toShards(shards, enc); err != nil {
			return err
		}
	}
	return nil
}

func (c *Codec) Decode(shard_paths []string, outpath string) error {
	var err error
	// Open shards.
	shards := make([]*os.File, len(shard_paths))
	for i,path := range shard_paths {
		shards[i], err = io.OpenFile(path)	
		if err != nil {
			return err
		}
		defer shards[i].Close()
	}

	// Read metadata from shards.
	n, padding, mat, err := metaFromShards(shards)
	chunk_size := int64(n)*CHUNK_SIZE

	// Create out file for decoded data.
	f, err := io.CreateFile(outpath)
	if err != nil {
		return err
	}
	defer f.Close()

	stop:
	for cnt:=0;;cnt++{
		// Read chunk of sharded data.
		chunk := make([][]byte, len(shards))
		for i,shard := range shards {
			chunk[i], err = io.ReadFrom(shard, chunk_size)
			if err != nil {
				return err
			}
			if len(chunk[i]) == 0 { // EOF
				break stop
			}
		}

		// Decode chunk.
		dec, err := c.pu.Decode(mat, chunk)
		if err != nil {
			return err
		}

		if cnt == 0 {
			dec = dec[padding:]
		}

		// Remove padding from first decoded chunk.
		// Write chunk to the output file.
		if err := io.WriteTo(f, dec); err != nil {
			return err
		}
	}
	return nil
}
