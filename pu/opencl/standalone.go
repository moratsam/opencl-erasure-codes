package opencl

import (
	"fmt"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
	"golang.org/x/xerrors"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

type OpenCLPU struct {
	context			*cl.Context	
	queue				*cl.CommandQueue
	kernel_decode	*cl.Kernel
	kernel_encode	*cl.Kernel

	exp_table	[]byte
	log_table	[]byte
}

func NewOpenCLPU(n int) (*OpenCLPU, error) {
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
	device := devices[0]
	printDeviceInfo(device)

	// Create device context & command queue.
	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		return nil, u.WrapErr("create context", err)
	}
	queue, err := context.CreateCommandQueue(device, 0)
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

	// Create encode kernel.
	kernel_encode, err := createKernel("encode", n, pu.context)
	if err != nil {
		return nil, u.WrapErr("create encode kernel", err)
	}
	pu.kernel_encode = kernel_encode

	return pu, nil
}

func (c *OpenCLPU) Decode(mat, data [][]byte) ([]byte, error) {
	n := len(mat[0])
	n_words := len(data[0])

	// Create kernel with correct n.
	kernel_decode, err := createKernel("decode", n, c.context)
	if err != nil {
		return nil, u.WrapErr("create decode kernel", err)
	}
	c.kernel_decode = kernel_decode

	// Potentially pad the data to be a multiple of local_dim1.
	padding := (local_dim1 - (n_words%local_dim1)) % local_dim1
	padded_n_words := padding+n_words

	// Flatten the inverted cauchy submatrix by appending rows.
	flat_mat := make([]byte, 0, n*n)
	for i:=0; i<n; i++ {
		flat_mat = append(flat_mat, mat[i]...)
	}

	// Flatten the enc data from shards by appending rows (one shard after another).
	flat_data := make([]byte, 0, n*padded_n_words)
	output := make([]byte, n*padded_n_words)
	
	for i:=0; i<n; i++ {
		flat_data = append(flat_data, data[i]...)
		flat_data = append(flat_data, make([]byte, padding)...)
	}

	if err := c.runKernel("decode", flat_mat, flat_data, output, []int{n, padded_n_words}, []int{n, local_dim1}); err != nil{
		return nil, u.WrapErr("run decode kernel", err)
	}

	return output[:n*n_words], nil
}

func (c *OpenCLPU) Encode(mat [][]byte, data []byte) ([][]byte, error) {
	n := len(mat[0])
	k := len(mat)-n
	n_words := len(data)/n

	// Flatten the cauchy matrix by appending rows.
	flat_mat := make([]byte, 0, (n+k)*n)
	for i:=0; i<n+k; i++ {
		for j:=0; j<n; j++ {
			flat_mat = append(flat_mat, mat[i][j])
		}
	}

	// Potentially pad the data to be a multiple of local_dim1.
	padding := (local_dim1 - (n_words%local_dim1)) % local_dim1
	padded_n_words := padding+n_words
	data = append(data, make([]byte, padding)...)

	// Allocate go-side storage for loading the output from the OpenCL program.
	output := make([]byte, (n+k)*(padded_n_words))

	if err := c.runKernel("encode", flat_mat, data, output, []int{n+k, padded_n_words}, []int{n+k, local_dim1}); err != nil{
		return nil, u.WrapErr("run encode kernel", err)
	}

	// Transform output into shard format.
	enc := make([][]byte, n+k)
	for i := range enc {
		enc[i] = output[i*padded_n_words:i*padded_n_words+n_words]
	}

	return enc, nil
}

func (c *OpenCLPU) runKernel(kernel_name string, mat, data, output []byte, global_work_size, local_work_size []int) error {
	byte_size := int(unsafe.Sizeof(byte(1)))
	var kernel *cl.Kernel
	if kernel_name == "decode" {
		kernel = c.kernel_decode
	} else {
		kernel = c.kernel_encode
	}

	// Enqueue input buffers.
	buf_exp_table, err := enqueueArr(c.exp_table, c.context, c.queue)
	if err != nil {
		return u.WrapErr("enqueue exp_table", err)
	}
	defer buf_exp_table.Release()
	buf_log_table, err := enqueueArr(c.log_table, c.context, c.queue)
	if err != nil {
		return u.WrapErr("enqueue log_table", err)
	}
	defer buf_log_table.Release()
	buf_mat, err := enqueueArr(mat, c.context, c.queue)
	if err != nil {
		return u.WrapErr("enqueue mat", err)
	}
	defer buf_mat.Release()
	buf_data, err := enqueueArr(data, c.context, c.queue)
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
	if err := kernel.SetArgs(buf_exp_table, buf_log_table, buf_mat, buf_data, buf_output); err != nil {
		return u.WrapErr("set args", err)
	}

	// Enqueue kernel.
	if _, err := c.queue.EnqueueNDRangeKernel(kernel, nil, global_work_size, local_work_size, nil); err != nil {
		return u.WrapErr("enqueue kernel", err)
	}

	// Block until queue is finished.
	if err := c.queue.Finish(); err != nil {
		return u.WrapErr("waiting to finish kernel", err)
	}

	// Copy data from OpenCL's output buffer to the go output array.
	outputPtr := unsafe.Pointer(&output[0])
	if _, err := c.queue.EnqueueReadBuffer(buf_output, true, 0, byte_size*len(output), outputPtr, nil); err != nil {
		return u.WrapErr("reading data from buffer", err)
	}

	return nil
}
