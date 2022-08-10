package opencl

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"

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
	local_dim1 int = 32
)

func createKernel(name string, n int, context *cl.Context) (*cl.Kernel, error) {
	var kernel_source string
	if name == "decode" {
		kernel_source = util_source + kernel_decode_source
	} else {
		kernel_source = util_source + kernel_encode_source
	}

	program, err := context.CreateProgramWithSource([]string{kernel_source})
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

func enqueueArr(arr []byte, context *cl.Context, queue *cl.CommandQueue) (*cl.MemObject, error) {
	elem_size := int(unsafe.Sizeof(arr[0]))
	ptr := unsafe.Pointer(&arr[0])
	buffer, err := context.CreateEmptyBuffer(cl.MemReadOnly, elem_size*len(arr))
	if err != nil {
		return nil, u.WrapErr("create buffer", err)
	}
	_, err = queue.EnqueueWriteBuffer(buffer, true, 0, elem_size*len(arr), ptr, nil)
	if err != nil {
		return nil, u.WrapErr("enqueue buffer", err)
	}

	return buffer, nil
}

func printDeviceInfo(device *cl.Device) {
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
		fmt.Println("\t", i.name, ":", i.value)
	}
}
