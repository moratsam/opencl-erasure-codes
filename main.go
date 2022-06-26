package main

import (
	"strconv"

	"github.com/moratsam/opencl-erasure-codes/codec"
	vl "github.com/moratsam/opencl-erasure-codes/pu/vanilla"
)

func main(){
	var err error
	k, n := 13, 7
	inpath := "in"
	outpath := "out"

	pu := vl.NewVanillaPU()
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
