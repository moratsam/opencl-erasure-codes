package opencl

import (
	"context"
	"fmt"

	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"
	"golang.org/x/xerrors"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

type Streamer struct {
	ix				int
	token_pool	chan struct{}

	device			*cl.Device
	context			*cl.Context	
	queue_kernel	*cl.CommandQueue // Queue over which kernel commands are sent.
	queue_read		*cl.CommandQueue // Queue over which read commands are sent.
	queue_write		*cl.CommandQueue // Queue over which write commands are sent.
	kernel 			*cl.Kernel

	// When decoding, this is the input channel,
	// when encoding, this is the output channel.
	c_2d	chan [][]byte
	// When decoding, this is the output channel,
	// when encoding, this is the input channel.
	c_1d	chan []byte

	cl_buf_exp_table	*cl.MemObject
	cl_buf_log_table	*cl.MemObject
	cl_buf_mat 			*cl.MemObject

	pip *pipeline.Pipeline
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
	printDeviceInfo(device)

	// Create device context & command queue.
	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		return nil, u.WrapErr("create context", err)
	}

	return &Streamer{ context: context, device: device }, nil
}

func (s *Streamer) InitDecoder(mat [][]byte) (chan []byte, error) {
	if err := s.init(mat, "decode"); err != nil {
		return nil, u.WrapErr("init decode", err)
	}
	go s.runDecode(context.Background(), local_dim1, len(mat)) // Spawn decoder routine.
	return s.c_1d, nil // Return channel over which the decoded data will be returned.
}

func (s *Streamer) InitEncoder(mat [][]byte) (chan [][]byte, error) {
	n := len(mat[0])
	k := len(mat) - n
	if err := s.init(mat, "encode"); err != nil {
		return nil, u.WrapErr("init encode", err)
	}
	go s.runEncode(context.Background(), local_dim1, n, k)// Spawn encoder routine.
	return s.c_2d, nil // Return channel over which the encoded data will be returned.
}

func (s *Streamer) Decode(chunk [][]byte) {
	s.c_2d <- chunk	
}

func (s *Streamer) Encode(chunk []byte) {
	s.c_1d <- chunk
}

func (s *Streamer) init(mat [][]byte, kernel_name string) error {
	n := len(mat[0])
	// Create kernel.
	kernel, err := createKernel(kernel_name, n, s.context)
	if err != nil {
		return u.WrapErr("create kernel", err)
	}
	s.kernel = kernel

	// Create queues.
	queue_kernel, err := s.context.CreateCommandQueue(s.device, 0)
	if err != nil {
		return u.WrapErr("create proc command queue", err)
	}
	s.queue_kernel = queue_kernel
	queue_read, err := s.context.CreateCommandQueue(s.device, 0)
	if err != nil {
		return u.WrapErr("create read command queue", err)
	}
	s.queue_read = queue_read
	queue_write, err := s.context.CreateCommandQueue(s.device, 0)
	if err != nil {
		return u.WrapErr("create write command queue", err)
	}
	s.queue_write = queue_write

	// Get exp and log tables for faster multiplication and division.
	exp_table, log_table := u.GetTables()

	// Enqueue & Set the constant kernel args (exp_table, log_table and matrix).
	buf_exp_table, err := enqueueArr(exp_table, s.context, s.queue_write)
	if err != nil {
		return u.WrapErr("enqueue exp_table", err)
	}
	buf_log_table, err :=enqueueArr(log_table, s.context, s.queue_write)
	if err != nil {
		return u.WrapErr("enqueue log_table", err)
	}
	flat_mat := make([]byte, 0, n*len(mat[0]))
	for i:=0; i<n; i++ {
		flat_mat = append(flat_mat, mat[i][:]...)
	}
	buf_mat, err := enqueueArr(flat_mat, s.context, s.queue_write)
	if err != nil {
		return u.WrapErr("enqueue mat", err)
	}
	if err := kernel.SetArg(0, buf_exp_table); err != nil {
		return u.WrapErr("set arg exp_table", err)
	}
	if err := kernel.SetArg(1, buf_log_table); err != nil {
		return u.WrapErr("set arg log_table", err)
	}
	if err := kernel.SetArg(2, buf_mat); err != nil {
		return u.WrapErr("set arg mat", err)
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
	s.c_2d = make(chan [][]byte, 1)
	s.c_1d = make(chan []byte, 1)

	return nil
}
