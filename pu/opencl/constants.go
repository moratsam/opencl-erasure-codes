package opencl

import _ "embed"

var (
	//go:embed util.cl
	util_source string
	//go:embed kernel_encode.cl
	kernel_encode_source string
	//go:embed kernel_decode.cl
	kernel_decode_source string
)

const (
	local_dim1 int = 32
)
