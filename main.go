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

	fmt.Println("correcto i[[69 108 93 100 193] [230 16 122 223 254] [163 124 89 54 167] [152 77 35 64 54] [44 70 78 127 160] [199 230 22 181 15] [120 98 42 21 59] [248 176 249 104 228] [69 212 3 69 187] [211 51 216 78 2]]")

	var err error
	k, n := 3, 7
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

	return

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
