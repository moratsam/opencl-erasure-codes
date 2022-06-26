package opencl

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
	"golang.org/x/xerrors"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

var (
	//go:embed util.cl
	util string
	//go:embed kernel_encode.cl
	kernel_encode string
	//go:embed kernel_decode.cl
	kernel_decode string
)

type OpenCLPU struct {
	context	*cl.Context	
	queue		*cl.CommandQueue

	exp_table	[]byte
	log_table	[]byte
}

func NewOpenCLPU() (*OpenCLPU, error) {
	// Get platforms.
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, u.WrapErr("get platforms", err)
	}
	fmt.Println("Using platform: " + platforms[0].Name(), ", profile: ", platforms[0].Profile(), ", with version: ", platforms[0].Version())

	// Get devices.
	devices, err := platforms[0].GetDevices(cl.DeviceTypeAll)
	if err != nil {
		return nil, u.WrapErr("get devices", err)
	}
	if len(devices) == 0 {
		return nil, u.WrapErr("", xerrors.New("GetDevices returned 0 devices"))
	}
	fmt.Println("Using: " + devices[0].Name(), ", type: ", devices[0].Type().String(), ", with openclc version: ", devices[0].OpenCLCVersion())

	// Create device context & command queue.
	context, err := cl.CreateContext([]*cl.Device{devices[0]})
	if err != nil {
		return nil, u.WrapErr("create context", err)
	}
	queue, err := context.CreateCommandQueue(devices[0], 0)
	if err != nil {
		return nil, u.WrapErr("create command queue", err)
	}

	// Get exp and log tables for faster multiplication and division.
	exp_table, log_table := u.GetTables()
	return &OpenCLPU{context, queue, exp_table, log_table}, nil
}

func (c *OpenCLPU) Decode(mat, enc [][]byte) ([]byte, error) {
	return nil, nil
}

func (c *OpenCLPU) Encode(mat [][]byte, data[]byte) ([][]byte, error) {
	fmt.Println("\n\n\n")
	n := byte(len(mat[0]))
	k := byte(len(mat))-n
	n_words := len(data)/int(n)
	fmt.Println(len(data))

	fmt.Println("mat", mat)
	// Flatten the cauchy matrix by appending rows.
	flat_mat := make([]byte, 0, int(n+k)*int(n))
	for i:=0; i<int(n+k); i++ {
		for j:=0; j<int(n); j++ {
			flat_mat = append(flat_mat, mat[i][j])
		}
	}
	fmt.Println("flat mat", flat_mat)
	fmt.Println("data", data)

	// Allocate go-side storage for loading the output from the OpenCL program.
	output := make([]byte, int(n+k)*n_words)

	// Enqueue input buffers.
	buf_exp_table, err := c.enqueueArr(c.exp_table)
	if err != nil {
		return nil, u.WrapErr("enqueue exp_table", err)
	}
	buf_log_table, err := c.enqueueArr(c.log_table)
	if err != nil {
		return nil, u.WrapErr("enqueue log_table", err)
	}
	buf_mat, err := c.enqueueArr(flat_mat)
	if err != nil {
		return nil, u.WrapErr("enqueue mat", err)
	}
	buf_data, err := c.enqueueArr(data)
	if err != nil {
		return nil, u.WrapErr("enqueue data", err)
	}
	fmt.Println("len output", len(output))
	//buf_n, err := c.enqueueByte(n)
	if err != nil {
		return nil, u.WrapErr("enqueue n", err)
	}

	// Create output buffer.
	byte_size := int(unsafe.Sizeof(n))
	buf_output, err := c.context.CreateEmptyBuffer(cl.MemReadOnly, byte_size*len(output))
	if err != nil {
		return nil, u.WrapErr("create output buffer", err)
	}

	// Create kernel.
	kernel, err := c.createKernel("encode")
	if err != nil {
		return nil, u.WrapErr("create kernel", err)
	}

	// Set kernel args.
	if err := kernel.SetArgs(buf_exp_table, buf_log_table, buf_mat, buf_data, n, buf_output); err != nil {
		return nil, u.WrapErr("set args", err)
	}

	// Enqueue kernel.
	if _, err := c.queue.EnqueueNDRangeKernel(kernel, nil, []int{int(n+k), n_words}, []int{1, 1}, nil); err != nil {
		return nil, u.WrapErr("enqueue kernel", err)
	}

	// Block until queue is finished.
	if err := c.queue.Finish(); err != nil {
		return nil, u.WrapErr("enqueue kernel", err)
	}

	// Copy data from OpenCL's output buffer to the go output array.
	outputPtr := unsafe.Pointer(&output[0])
	if _, err := c.queue.EnqueueReadBuffer(buf_output, true, 0, byte_size*len(output), outputPtr, nil); err != nil {
		return nil, u.WrapErr("reading data from buffer", err)
	}

	// Transform output into shard format.
	fmt.Println("gpu output", output)
	enc := make([][]byte, int(n+k))
	fmt.Println("n+k", n+k)
	for i := range enc {
		enc[i] = make([]byte, n_words)
	}
	fmt.Println("enc dim:", len(enc), len(enc[0]))
	for i := range enc {
		for word_ix:=0; word_ix<n_words; word_ix++ {
			enc[i][word_ix] = output[i*n_words + word_ix]
		}
	}
	fmt.Println("sharded", enc)

	return enc, nil
}

func (c *OpenCLPU) createKernel(name string) (*cl.Kernel, error) {
	var kernel_source string
	if name == "decode" {
		kernel_source = util + kernel_decode
	} else {
		kernel_source = util + kernel_encode
	}

	program, err := c.context.CreateProgramWithSource([]string{kernel_source})
	if err != nil {
		return nil, u.WrapErr("create program", err)
	}

	if err := program.BuildProgram(nil, ""); err != nil {
		return nil, u.WrapErr("build program", err)
	}
	
	kernel, err := program.CreateKernel(name)
	if err != nil {
		return nil, u.WrapErr("create kernel", err)
	}

	return kernel, nil
}

func (c *OpenCLPU) enqueueArr(arr []byte) (*cl.MemObject, error) {
	elem_size := int(unsafe.Sizeof(arr[0]))
	ptr := unsafe.Pointer(&arr[0])
	buffer, err := c.context.CreateEmptyBuffer(cl.MemReadOnly, elem_size*len(arr))
	if err != nil {
		return nil, u.WrapErr("create buffer", err)
	}
	_, err = c.queue.EnqueueWriteBuffer(buffer, true, 0, elem_size*len(arr), ptr, nil)
	if err != nil {
		return nil, u.WrapErr("enqueue buffer", err)
	}

	return buffer, nil
}

func (c *OpenCLPU) enqueueByte(b byte) (*cl.MemObject, error) {
	elem_size := int(unsafe.Sizeof(b))
	fmt.Println("byte size", elem_size)
	ptr := unsafe.Pointer(&b)
	buffer, err := c.context.CreateEmptyBuffer(cl.MemReadOnly, elem_size*1)
	if err != nil {
		return nil, u.WrapErr("create buffer", err)
	}
	_, err = c.queue.EnqueueWriteBuffer(buffer, true, 0, elem_size*1, ptr, nil)
	if err != nil {
		return nil, u.WrapErr("enqueue buffer", err)
	}

	return buffer, nil
}
