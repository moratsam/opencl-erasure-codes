package io

import (
	"os"
	"io"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

func CreateFile(filepath string) (*os.File, error) {
	return os.Create(filepath)
}

func OpenFile(filepath string) (*os.File, error) {
	return os.Open(filepath)	
}

func FileSize(filepath string) (int64, error) {
	fi, err := os.Stat(filepath)
	if err != nil {
		return 0, u.WrapErr("get stat", err)
	}
	return fi.Size(), nil
}

func ReadFrom(f *os.File, chunk_size int64) ([]byte, error) {
	chunk := make([]byte, chunk_size)
	count, err := f.Read(chunk)
	if err != nil {
		if err == io.EOF {
			return make([]byte, 0), nil
		}
		return nil, u.WrapErr("read", err)
	}
	return chunk[:count], nil
}

func WriteTo(f *os.File, chunk []byte) error {
	_, err := f.Write(chunk)
	return err
}
