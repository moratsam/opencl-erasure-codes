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
	util_source string
	//go:embed kernel_encode.cl
	kernel_encode_source string
	//go:embed kernel_decode.cl
	kernel_decode_source string
)

const local_dim1 int = 32;

type OpenCLPU struct {
	context			*cl.Context	
	queue				*cl.CommandQueue
	kernel_decode	*cl.Kernel
	kernel_encode	*cl.Kernel

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
	/*
	device := devices[0]
	fmt.Println("i\nUsing device:")

	info := []struct{name string; value interface{}}{
		{"name", device.Name()},
		{"type", device.Type()},
		{"profile", device.Profile()},
		{"vendor", device.Vendor()},
		{"version", device.Version()},
		{"driver version", device.DriverVersion()},
		{"openCL C version", device.OpenCLCVersion()},
		{"address bits", device.AddressBits()},
		{"little endian", device.EndianLittle()},
		{"extensions", device.Extensions()},
		{"global mem cache size", device.GlobalMemCacheSize()},
		{"global mem size", device.GlobalMemSize()},
		{"local mem size", device.LocalMemSize()},
		{"max clock frequency", device.MaxClockFrequency()},
		{"max compute units", device.MaxComputeUnits()},
		{"max constant buffer size", device.MaxConstantBufferSize()},
		{"max mem alloc size", device.MaxMemAllocSize()},
		{"max parameter size", device.MaxParameterSize()},
		{"max work group size", device.MaxWorkGroupSize()},
		{"max workitem dimensions", device.MaxWorkItemDimensions()},
		{"max workitem sizes", device.MaxWorkItemSizes()},
		{"native vector width char", device.NativeVectorWidthChar()},
		{"native vector width double", device.NativeVectorWidthDouble()},
		{"native vector width float", device.NativeVectorWidthFloat()},
		{"native vector width int", device.NativeVectorWidthInt()},
	}
	for _,i := range info {
		func(name string, out interface{}) {
			switch out.(type) {
				default:
					fmt.Println("\t", name, ":", out)
			}
		}(i.name, i.value)
	}
	*/
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

	pu := &OpenCLPU{
		context: 	context,
		queue:		queue,
		exp_table:	exp_table,
		log_table:	log_table,
	}

	// Create kernels.
	kernel_decode, err := pu.createKernel("decode")
	if err != nil {
		return nil, u.WrapErr("create decode kernel", err)
	}
	kernel_encode, err := pu.createKernel("encode")
	if err != nil {
		return nil, u.WrapErr("create encode kernel", err)
	}
	pu.kernel_decode = kernel_decode
	pu.kernel_encode = kernel_encode

	return pu, nil
}

func (c *OpenCLPU) Decode(mat, data [][]byte) ([]byte, error) {
	//fmt.Println("\n\n\n")
	n := len(mat[0])
	n_words := len(data[0])

	// Potentially pad the data to be a multiple of local_dim1.
	padding := (local_dim1 - (n_words%local_dim1)) % local_dim1
	//fmt.Println("padding", padding)
	padded_n_words := padding+n_words
	//fmt.Println("padded n words", padded_n_words)

	//fmt.Println("mat", mat)
	// Flatten the inverted cauchy submatrix by appending rows.
	flat_mat := make([]byte, 0, n*n)
	for i:=0; i<n; i++ {
		for j:=0; j<n; j++ {
			flat_mat = append(flat_mat, mat[i][j])
		}
	}
	//fmt.Println("flat mat", flat_mat)

	//fmt.Println("data", data)
	// Flatten the enc data from shards by appending rows (one shard after another).
	flat_data := make([]byte, 0, n*padded_n_words)
	for i:=0; i<n; i++ {
		flat_data = append(flat_data, data[i]...)
		flat_data = append(flat_data, make([]byte, padding)...)
	}
	//fmt.Println("flat_data", flat_data)

	// Allocate go-side storage for loading the output from the OpenCL program.
	output := make([]byte, n*padded_n_words)
	//fmt.Println("len output", len(output))

	if err := c.runKernel("decode", flat_mat, flat_data, output, byte(n), []int{n, padded_n_words}, []int{n, local_dim1}); err != nil{
		return nil, u.WrapErr("enqueue kernel", err)
	}

	//fmt.Println("output", output)

	return output[:n*n_words], nil
}

func (c *OpenCLPU) Encode(mat [][]byte, data[]byte) ([][]byte, error) {
	//fmt.Println("\n\n\n")
	n := len(mat[0])
	k := len(mat)-n
	n_words := len(data)/n
	//fmt.Println(len(data))

	//fmt.Println("mat", mat)
	// Flatten the cauchy matrix by appending rows.
	flat_mat := make([]byte, 0, (n+k)*n)
	for i:=0; i<n+k; i++ {
		for j:=0; j<n; j++ {
			flat_mat = append(flat_mat, mat[i][j])
		}
	}
	//fmt.Println("flat mat", flat_mat)
	//fmt.Println("data", data)

	// Potentially pad the data to be a multiple of local_dim1.
	padding := (local_dim1 - (n_words%local_dim1)) % local_dim1
	//fmt.Println("padding", padding)
	padded_n_words := padding+n_words
	//fmt.Println("padded n words", padded_n_words)

	// Allocate go-side storage for loading the output from the OpenCL program.
	output := make([]byte, (n+k)*(padded_n_words))
	//fmt.Println("len output", len(output))

	if err := c.runKernel("encode", flat_mat, data, output, byte(n), []int{n+k, padded_n_words}, []int{n+k, local_dim1}); err != nil{
		return nil, u.WrapErr("enqueue kernel", err)
	}

	// Transform output into shard format.
	//fmt.Println("gpu output", output)
	enc := make([][]byte, n+k)
	//fmt.Println("n+k", n+k)
	//fmt.Println("enc dim:", len(enc), len(enc[0]))
	for i := range enc {
		enc[i] = output[i*padded_n_words:i*padded_n_words+n_words]
	}
	//fmt.Println("sharded", enc)

	return enc, nil
}

func (c *OpenCLPU) runKernel(kernel_name string, mat, data, output []byte, n byte, global_work_size, local_work_size []int) error {
	byte_size := int(unsafe.Sizeof(n))
	var kernel *cl.Kernel
	if kernel_name == "decode" {
		kernel = c.kernel_decode
	} else {
		kernel = c.kernel_encode
	}

	// Enqueue input buffers.
	buf_exp_table, err := c.enqueueArr(c.exp_table)
	if err != nil {
		return u.WrapErr("enqueue exp_table", err)
	}
	defer buf_exp_table.Release()
	buf_log_table, err := c.enqueueArr(c.log_table)
	if err != nil {
		return u.WrapErr("enqueue log_table", err)
	}
	defer buf_log_table.Release()
	buf_mat, err := c.enqueueArr(mat)
	if err != nil {
		return u.WrapErr("enqueue mat", err)
	}
	defer buf_mat.Release()
	buf_data, err := c.enqueueArr(data)
	if err != nil {
		return u.WrapErr("enqueue data", err)
	}
	defer buf_data.Release()

	// Create output buffer.
	buf_output, err := c.context.CreateEmptyBuffer(cl.MemReadOnly, byte_size*len(output))
	if err != nil {
		return u.WrapErr("create output buffer", err)
	}
	defer buf_output.Release()

	// Set kernel args.
	if err := kernel.SetArgs(buf_exp_table, buf_log_table, buf_mat, buf_data, n, buf_output); err != nil {
		return u.WrapErr("set args", err)
	}

	// Enqueue kernel.
	if _, err := c.queue.EnqueueNDRangeKernel(kernel, nil, global_work_size, local_work_size, nil); err != nil {
		return u.WrapErr("enqueue kernel", err)
	}

	// Block until queue is finished.
	if err := c.queue.Finish(); err != nil {
		return u.WrapErr("enqueue kernel", err)
	}

	// Copy data from OpenCL's output buffer to the go output array.
	outputPtr := unsafe.Pointer(&output[0])
	if _, err := c.queue.EnqueueReadBuffer(buf_output, true, 0, byte_size*len(output), outputPtr, nil); err != nil {
		return u.WrapErr("reading data from buffer", err)
	}

	return nil
}

func (c *OpenCLPU) createKernel(name string) (*cl.Kernel, error) {
	var kernel_source string
	if name == "decode" {
		kernel_source = util_source + kernel_decode_source
	} else {
		kernel_source = util_source + kernel_encode_source
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
