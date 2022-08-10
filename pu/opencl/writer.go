package opencl

import (
	"context"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

type writer struct {
	dev_context	*cl.Context
	queue 		*cl.CommandQueue
}

func newWriter(dev_context *cl.Context, queue *cl.CommandQueue) *writer {
	return &writer{ dev_context, queue }
}

// This step in the processing pipeline copies the input buffer from host onto the device.
func (w *writer) Process(_ context.Context, payload pipeline.Payload) (pipeline.Payload, error) {
	p := payload.(*streamerPayload)

	byte_size := int(unsafe.Sizeof(p.host_in[0]))

	// Create write buffer.
	cl_buf_out, err := w.dev_context.CreateEmptyBuffer(cl.MemWriteOnly, byte_size*len(p.host_out))
	if err != nil {
		return nil, u.WrapErr("create cl_buf_out", err)
	}
	p.cl_buf_out = cl_buf_out

	// Create read buffer.
	cl_buf_in, err := w.dev_context.CreateEmptyBuffer(cl.MemReadOnly, byte_size*len(p.host_in))
	if err != nil {
		return nil, u.WrapErr("create cl_buf_in", err)
	}
	p.cl_buf_in = cl_buf_in

	// Write input data to device.
	ptr := unsafe.Pointer(&p.host_in[0])
	_, err = w.queue.EnqueueWriteBuffer(cl_buf_in, true, 0, byte_size*len(p.host_in), ptr, nil)
	if err != nil {
		return nil, u.WrapErr("enqueue cl_buf_in", err)
	}

	return p, nil
}
