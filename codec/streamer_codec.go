package codec

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/moratsam/opencl-erasure-codes/io"
	proc_unit "github.com/moratsam/opencl-erasure-codes/pu"
	u "github.com/moratsam/opencl-erasure-codes/util"
)

type StreamerCodec struct{
	pu	proc_unit.StreamerPU
}
func NewStreamerCodec(pu proc_unit.StreamerPU) *StreamerCodec {
	return &StreamerCodec{pu}
}

func (c *StreamerCodec) Encode(k, n byte, filepath string) error {
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

	// Receive channel over which the encoded data will be sent by the encoding streamer.
	c_data, err := c.pu.InitEncoder(mat)
	if err != nil {
		return err
	}

	// Separate routine receives encoded data from the streamer and writes it to the shards.
	c_num := make(chan int, 1)
	c_done := make(chan struct{}, 1)
	go func(){
		cnt := 0
		max_cnt := -1
		for {
			select {
			case max_cnt = <- c_num:
			case enc := <- c_data:
				// Write chunk to the shards.
				if err := toShards(shards, enc); err != nil {
					panic(u.WrapErr("enc file write", err))
					return
				}
				cnt++
			}
			if cnt == max_cnt {
				c_done <- struct{}{}
				return
			}
		}
	}()

	var now, enc time.Time
	now = time.Now()
	var cnt int
	for cnt=0;;cnt++{
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
		c.pu.Encode(chunk)
	}
	c_num <- cnt
	<-c_done
	
	enc = enc.Add(time.Since(now))
	fmt.Println("\nencode time:", enc.Sub(time.Time{}))

	return nil
}

func (c *StreamerCodec) Decode(shard_paths []string, outpath string) error {
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

	// Receive channel over which the decoded data will be sent by the decoding streamer.
	c_data, err := c.pu.InitDecoder(mat)
	if err != nil {
		return err
	}

	// Separate routine receives decoded data from the streamer and writes it to a file.
	c_num := make(chan int, 1)
	c_done := make(chan struct{}, 1)
	go func(){
		cnt := 0
		max_cnt := -1
		for {
			select {
			case max_cnt = <- c_num:
			case dec := <- c_data:
				// Remove padding from first decoded chunk.
				if cnt == 0 {
					dec = dec[padding:]
				}
				// Write chunk to the output file.
				if err := io.WriteTo(f, dec); err != nil {
					panic(u.WrapErr("dec file write", err))
					return
				}
				cnt++
			}
			if cnt == max_cnt {
				c_done <- struct{}{}
				return
			}
		}
	}()

	var now, dec time.Time
	now = time.Now()
	var cnt int
	stop:
	for cnt=0;;cnt++{
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
		c.pu.Decode(chunk)
	}
	c_num <- cnt
	<-c_done
	
	dec=dec.Add(time.Since(now))
	fmt.Println("\ndecode time:", dec.Sub(time.Time{}))

	return nil
}
