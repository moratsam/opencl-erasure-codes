package opencl

import (
	"context"

	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

type kernelRunner struct {
	kernel 	*cl.Kernel
	queue		*cl.CommandQueue
}

func newKernelRunner(kernel *cl.Kernel, queue *cl.CommandQueue) *kernelRunner {
	return &kernelRunner{kernel, queue }
}

func (k *kernelRunner) Process(_ context.Context, payload pipeline.Payload) (pipeline.Payload, error) {
	p := payload.(*streamerPayload)

	// Set kernel args.
	if err := k.kernel.SetArg(3, p.cl_buf_in); err != nil {
		return nil, u.WrapErr("set args cl_buf_in", err)
	}

	if err := k.kernel.SetArg(4, p.cl_buf_out); err != nil {
		return nil, u.WrapErr("set args cl_buf_in", err)
	}

	// Enqueue kernel.
	if _, err := k.queue.EnqueueNDRangeKernel(k.kernel, nil, p.global_work_size, p.local_work_size, nil); err != nil {
		return nil, u.WrapErr("enqueue kernel", err)
	}

	// Block until queue is finished.
	if err := k.queue.Finish(); err != nil {
		return nil, u.WrapErr("kernel finish", err)
	}

	return p, nil
}
