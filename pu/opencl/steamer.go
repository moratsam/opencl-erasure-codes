package opencl

import (
	"context"
	_ "embed"
	"fmt"
	"unsafe"
	"time"

	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"
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

const (
	local_dim1	int = 32
)

type Streamer struct {
	ix				int
	token_pool	chan struct{}

	device			*cl.Device
	context			*cl.Context	
	queue_kernel		*cl.CommandQueue
	queue_read		*cl.CommandQueue
	queue_write		*cl.CommandQueue
	kernel 			*cl.Kernel

	c_dec_in		chan [][]byte
	c_dec_out	chan []byte

	cl_buf_exp_table	*cl.MemObject
	cl_buf_log_table	*cl.MemObject
	cl_buf_mat 			*cl.MemObject

	pip *pipeline.Pipeline

	now, lmao, data_manipul_in, ker_total, enq_writ, enq_read, ker_proc time.Time
}

func (s *Streamer) InitDecoder(mat [][]byte) (chan []byte, error) {
	n := len(mat)
	// Create decode kernel.
	kernel, err := s.createKernel("decode", n)
	if err != nil {
		return nil, u.WrapErr("create decode kernel", err)
	}
	s.kernel = kernel

	// Create queues.
	queue_kernel, err := s.context.CreateCommandQueue(s.device, 0)
	if err != nil {
		return nil, u.WrapErr("create proc command queue", err)
	}
	s.queue_kernel = queue_kernel
	queue_read, err := s.context.CreateCommandQueue(s.device, 0)
	if err != nil {
		return nil, u.WrapErr("create read command queue", err)
	}
	s.queue_read = queue_read
	queue_write, err := s.context.CreateCommandQueue(s.device, 0)
	if err != nil {
		return nil, u.WrapErr("create writ command queue", err)
	}
	s.queue_write = queue_write

	// Get exp and log tables for faster multiplication and division.
	exp_table, log_table := u.GetTables()

	// Enqueue & Set the constant kernel args (exp_table, log_table and inv matrix).
	buf_exp_table, err := s.enqueueArr(exp_table)
	if err != nil {
		return nil, u.WrapErr("enqueue exp_table", err)
	}
	buf_log_table, err :=s.enqueueArr(log_table)
	if err != nil {
		return nil, u.WrapErr("enqueue log_table", err)
	}
	flat_mat := make([]byte, 0, n*n)
	for i:=0; i<n; i++ {
		flat_mat = append(flat_mat, mat[i][:]...)
	}
	buf_mat, err := s.enqueueArr(flat_mat)
	if err != nil {
		return nil, u.WrapErr("enqueue mat", err)
	}
	if err := kernel.SetArg(0, buf_exp_table); err != nil {
		return nil, u.WrapErr("set arg exp_table", err)
	}
	if err := kernel.SetArg(1, buf_log_table); err != nil {
		return nil, u.WrapErr("set arg log_table", err)
	}
	if err := kernel.SetArg(2, buf_mat); err != nil {
		return nil, u.WrapErr("set arg mat", err)
	}
	s.cl_buf_exp_table = buf_exp_table
	s.cl_buf_log_table = buf_log_table
	s.cl_buf_mat = buf_mat

	// Assemble pipeline.
	pipeline_cfg := pipelineConfig{
		dev_context:	s.context,
		kernel: 			s.kernel,
		queue_kernel:	s.queue_kernel,
		queue_read:		s.queue_read,
		queue_write:	s.queue_write,
	}
	s.pip = assemblePipeline(pipeline_cfg)

	// Create in&out chans.
	s.c_dec_in = make(chan [][]byte, 1)
	s.c_dec_out = make(chan []byte, 1)

	go s.runDecode(context.Background(), local_dim1, n)

	return s.c_dec_out, nil
}


func (s *Streamer) Decode(chunk [][]byte) {
	s.c_dec_in <- chunk	
}

func NewStreamerPU() (*Streamer, error) {
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
	/*
	fmt.Println("\nUsing device:")

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
	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		return nil, u.WrapErr("create context", err)
	}

	return &Streamer{ context: context, device: device }, nil
}

func (s *Streamer) createKernel(name string, n int) (*cl.Kernel, error) {
	var kernel_source string
	if name == "decode" {
		kernel_source = util_source + kernel_decode_source
	} else {
		kernel_source = util_source + kernel_encode_source
	}

	program, err := s.context.CreateProgramWithSource([]string{kernel_source})
	if err != nil {
		return nil, u.WrapErr("create program", err)
	}

	options := fmt.Sprintf("-DSIZE_N=%d -DMAX_LID1=%d", n, local_dim1)
	if err := program.BuildProgram(nil, options); err != nil {
		return nil, u.WrapErr("build program", err)
	}
	
	kernel, err := program.CreateKernel(name)
	if err != nil {
		return nil, u.WrapErr("create kernel", err)
	}

	return kernel, nil
}

func (s *Streamer) enqueueArr(arr []byte) (*cl.MemObject, error) {
	elem_size := int(unsafe.Sizeof(arr[0]))
	ptr := unsafe.Pointer(&arr[0])
	buffer, err := s.context.CreateEmptyBuffer(cl.MemReadOnly, elem_size*len(arr))
	if err != nil {
		return nil, u.WrapErr("create buffer", err)
	}
	_, err = s.queue_write.EnqueueWriteBuffer(buffer, true, 0, elem_size*len(arr), ptr, nil)
	if err != nil {
		return nil, u.WrapErr("enqueue buffer", err)
	}

	return buffer, nil
}

func (s *Streamer) Encode(_ []byte) {}