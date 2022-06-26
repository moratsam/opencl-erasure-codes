package opencl

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
	"golang.org/x/xerrors"

	u "github.com/moratsam/opencl-erasure-codes/utils"
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

func (c *OpenCLPU) enqueueArr(arr []byte) (*cl.MemObject, error) {
	elem_size := int(unsafe.Sizeof(arr[0]))
	arr_ptr := unsafe.Pointer(&arr[0])
	arr_buffer, err := c.context.CreateEmptyBuffer(cl.MemReadOnly, elem_size*len(arr))
	if err != nil {
		return nil, u.WrapErr("create buffer", err)
	}
	// TODO maybe manually release.
	//defer arr_buffer.Release()
	_, err = c.queue.EnqueueWriteBuffer(arr_buffer, true, 0, elem_size*len(arr), arr_ptr, nil)
	if err != nil {
		return nil, u.WrapErr("enqueue buffer", err)
	}

	return arr_buffer, nil
}
