package main

import (
	"fmt"
	"strconv"

	"github.com/moratsam/opencl-erasure-codes/codec"
	cl "github.com/moratsam/opencl-erasure-codes/pu/opencl"
	vl "github.com/moratsam/opencl-erasure-codes/pu/vanilla"
)

func main(){
	fmt.Println()

	var err error
	k, n := 3, 8
	inpath := "in"
	outpath := "out"
	
	cl.NewOpenCLPU()
	vl.NewVanillaPU()
	/*
	pu := vl.NewVanillaPU()
	*/
	pu, err := cl.NewOpenCLPU()
	check(err)

	codec := codec.NewCodec(pu)

	// Encode
	err = codec.Encode(byte(k), byte(n), inpath)
	check(err)

	// Decode
	shard_fnames := make([]string, int(n))
	for i := range shard_fnames {
		shard_fnames[i] = inpath + "_" + strconv.Itoa(i) + ".enc"
	}
	err = codec.Decode(shard_fnames, outpath)
	check(err)
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
