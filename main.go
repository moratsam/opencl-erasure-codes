package main

import (
	"fmt"
	"strconv"

	
	"github.com/moratsam/opencl-erasure-codes/io"
	"github.com/moratsam/opencl-erasure-codes/codec"
	cl "github.com/moratsam/opencl-erasure-codes/pu/opencl"
	vl "github.com/moratsam/opencl-erasure-codes/pu/vanilla"
)

func main(){
	fmt.Println("\n\n")
	var err error
	k, n := 3, 7
	inpath := "in"
	outpath := "out"
	
	//cl.NewOpenCLPU()
	vl.NewVanillaPU()
	/*
	pu := vl.NewVanillaPU()
	pu, err := cl.NewOpenCLPU()
	check(err)

	codec := codec.NewCodec(pu)
	*/

	pu, err := cl.NewStreamerPU()
	check(err)
	codec := codec.NewStreamerCodec(pu)


	/*
	// Encode
	err = codec.Encode(byte(k), byte(n), inpath)
	check(err)
	*/
	_ = k

	// Decode
	shard_fnames := make([]string, int(n))
	for i := range shard_fnames {
		shard_fnames[i] = inpath + "_" + strconv.Itoa(i) + ".enc"
	}
	err = codec.Decode(shard_fnames, outpath)
	check(err)

	checkByByte(inpath, outpath)
}

// Check that input&output files are identical.
func checkByByte(in, out string) {
	// Check file sizes match.
	in_size, err := io.FileSize(in)
	check(err)
	out_size, err := io.FileSize(out)
	check(err)
	if in_size != out_size {
		panic("in&out file sizes don't match")
	}

	fin, err := io.OpenFile(in)
	check(err)
	fout, err := io.OpenFile(out)
	check(err)

	//Check that the input & output files match, byte by byte.
	chunk_size := int64(1000000)
	for {
		in_chunk, err := io.ReadFrom(fin, chunk_size)
		check(err)
		out_chunk, err := io.ReadFrom(fout, chunk_size)
		check(err)
		if len(in_chunk) != len(out_chunk) {
			panic("in&out chunk sizes don't match")
		}
		for i := range in_chunk {
			if in_chunk[i] != out_chunk[i] {
				s := fmt.Sprintf("in&out chunks don't match at byte %d: %d != %d", i, in_chunk[i], out_chunk[i])
				panic(s)
			}
		}
		if len(in_chunk) == 0 { // EOF
			break
		}
		//fmt.Println("Looking good, king.", len(in_chunk))
	}
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
