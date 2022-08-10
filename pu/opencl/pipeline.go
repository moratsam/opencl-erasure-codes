package opencl

import (
	"context"

	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"

	u "github.com/moratsam/opencl-erasure-codes/util"
)

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
	source := &decSource{n: n, local_dim1: local_dim1, c: s.c_2d}
	sink := &decSink{c: s.c_1d}
	if err := s.pip.Process(ctx, source, sink); err != nil {
		panic(u.WrapErr("decoding process", err))
	}
	return nil
}

func (s *Streamer) runEncode(ctx context.Context, local_dim1, n, k int) error {
	source := &encSource{n: n, k: k, local_dim1: local_dim1, c: s.c_1d}
	sink := &encSink{n: n, k: k, c: s.c_2d}
	if err := s.pip.Process(ctx, source, sink); err != nil {
		panic(u.WrapErr("encoding process", err))
	}

	return nil
}

// Source of the decoding pipeline.
type decSource struct { 
	n 				int
	local_dim1	int
	c 				chan [][]byte // Channel over which data encoded sharded data is received.
}
func (s *decSource) Error()						error	{ return nil }
func (s *decSource) Next(_ context.Context)	bool	{ return true }
// The decoding source receives chunks of encoded sharded data and loads them into a payload.
func (s *decSource) Payload() pipeline.Payload {
	data := <- s.c
	n_words := len(data[0])

	// Potentially pad the data to be a multiple of local_dim1.
	padding := (s.local_dim1 - (n_words%s.local_dim1)) % s.local_dim1
	padded_n_words := padding+n_words

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

// Sink of the decoding pipeline.
type decSink struct { c chan []byte }
// The decoding sink makes a copy of the output data and sends it to the output channel.
func (s *decSink) Consume(_ context.Context, payload pipeline.Payload) error {
	p := payload.(*streamerPayload)

	out := make([]byte, p.n*p.n_words)
	copy(out, p.host_out)
	s.c <- out
	return nil
}

// Source of the encoding pipeline.
type encSource struct {
	k 				int
	n 				int
	local_dim1	int
	c 				chan []byte
}
func (s *encSource) Error()						error	{ return nil }
func (s *encSource) Next(_ context.Context)	bool	{ return true }
// The encoding source receives chunks of original data and loads them into a payload.
func (s *encSource) Payload() pipeline.Payload {
	data := <- s.c
	n_words := len(data)/s.n

	// Potentially pad the data to be a multiple of local_dim1.
	padding := (s.local_dim1 - (n_words%s.local_dim1)) % s.local_dim1
	padded_n_words := padding+n_words

	p := payloadPool.Get().(*streamerPayload)
	p.n = s.n
	p.n_words = n_words
	p.padded_n_words = padded_n_words
	p.global_work_size = []int{p.n+s.k, padded_n_words}
	p.local_work_size = []int{p.n+s.k, s.local_dim1}

	// In the first or last runs, allocate go-side storage buffers.
	if ! (cap(p.host_in) == 0 && len(p.host_in) == len(data)) {
		p.host_in = make([]byte, 0, len(data))
	}
	if len(p.host_out) < (p.n+s.k)*p.padded_n_words {
		p.host_out = make([]byte, (p.n+s.k)*p.padded_n_words)
	}
	p.host_in = append(p.host_in, data...)
	p.host_in = append(p.host_in, make([]byte, padding)...)

	return p
}

// Sink of the encoding pipeline.
type encSink struct { 
	k int
	n int
	c chan [][]byte
}
// The encoding sink makes a copy of the output data, formats it into shard format
// and sends it to the output channel.
func (s *encSink) Consume(_ context.Context, payload pipeline.Payload) error {
	p := payload.(*streamerPayload)

	out := make([]byte, (s.n+s.k)*p.padded_n_words)
	copy(out, p.host_out)
	enc := make([][]byte, s.n+s.k)
	for i := range enc {
		enc[i] = out[i*p.padded_n_words:i*p.padded_n_words+p.n_words]
	}
	s.c <- enc
	return nil
}
