package opencl

import (
	"context"
	"fmt"

	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

type decSource struct { 
	n int
	local_dim1 int
	c chan [][]byte
}
func (s *decSource) Error()	error		{ return nil }
func (s *decSource) Next(_ context.Context)		bool 		{ return true }
func (s *decSource) Payload() pipeline.Payload {
	data := <- s.c
	n_words := len(data[0])

	// Potentially pad the data to be a multiple of local_dim1.
	padding := (s.local_dim1 - (n_words%s.local_dim1)) % s.local_dim1
	//fmt.Println("padding", padding)
	padded_n_words := padding+n_words
	//fmt.Println("padded n words", padded_n_words)

	p := payloadPool.Get().(*streamerPayload)
	p.n = s.n
	p.n_words = n_words
	p.padded_n_words = padded_n_words
	p.global_work_size = []int{p.n, padded_n_words}
	p.local_work_size = []int{p.n, s.local_dim1}

	if ! (cap(p.host_in) == 0 && len(p.host_in) == p.n*padded_n_words) {
		p.host_in = make([]byte, 0, p.n*padded_n_words)
	}
	if ! (len(p.host_out) >= p.n*p.padded_n_words) {
		p.host_out = make([]byte, p.n*p.padded_n_words)
	}
	for i:=0; i<p.n; i++ {
		p.host_in = append(p.host_in, data[i]...)
		p.host_in = append(p.host_in, make([]byte, padding)...)
	}

	return p
}

type decSink struct { c chan []byte }
func (s *decSink) Consume(_ context.Context, payload pipeline.Payload) error {
	p := payload.(*streamerPayload)

	fmt.Println("consuming")
	out := make([]byte, p.n*p.n_words)
	copy(out, p.host_out)
	s.c <- out
	return nil
}

type pipelineConfig struct {
	global_work_size	int
	local_work_size 	int

	dev_context		*cl.Context
	kernel 			*cl.Kernel
	queue_kernel	*cl.CommandQueue
	queue_read		*cl.CommandQueue
	queue_write		*cl.CommandQueue
}

func assemblePipeline(cfg pipelineConfig) *pipeline.Pipeline {
	return pipeline.New(
		pipeline.DynamicWorkerPool(newWriter(cfg.dev_context, cfg.queue_write), 1),
		pipeline.FIFO(newKernelRunner(cfg.kernel, cfg.queue_kernel)),
		pipeline.DynamicWorkerPool(newReader(cfg.queue_read), 1),
	)
}

func (s *Streamer) runDecode(ctx context.Context, local_dim1, n int) error {
	sink := &decSink{ c: s.c_dec_out }
	err := s.pip.Process(ctx, &decSource{n: n, local_dim1: local_dim1, c: s.c_dec_in}, sink)
	if err != nil {
		panic(u.WrapErr("decoding process", err))
	}
	return err
}
