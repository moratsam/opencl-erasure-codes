package opencl

import (
	"context"
	"unsafe"

	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

type reader struct {
	queue *cl.CommandQueue
}

func newReader(queue *cl.CommandQueue) *reader {
	return &reader{ queue }
}

func (r *reader) Process(_ context.Context, payload pipeline.Payload) (pipeline.Payload, error) {
	p := payload.(*streamerPayload)

	byte_size := int(unsafe.Sizeof(p.host_in[0]))
	// Read output from device.
	ptr := unsafe.Pointer(&p.host_out[0])
	_, err := r.queue.EnqueueReadBuffer(p.cl_buf_out, true, 0, byte_size*len(p.host_out), ptr, nil)
	if err != nil {
		return nil, u.WrapErr("enqueue cl_buf_out", err)
	}

	return p, nil
}
