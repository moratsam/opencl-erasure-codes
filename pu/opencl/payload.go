package opencl

import (
	"sync"
	
	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"
)

var payloadPool = sync.Pool{
	New: func() interface{} {return new(streamerPayload) },
}

type streamerPayload struct {
	n					int
	n_words			int
	padded_n_words int
	global_work_size []int
	local_work_size []int
	host_in			[]byte
	host_out			[]byte
	cl_buf_in		*cl.MemObject
	cl_buf_out		*cl.MemObject
}

// Doesn't really clone, cloning isn't needed.
func (p *streamerPayload) Clone() pipeline.Payload {
	return payloadPool.Get().(*streamerPayload)
}

func (p *streamerPayload) MarkAsProcessed() {
	p.cl_buf_in.Release()
	p.cl_buf_in.Release()
	p.host_in = p.host_in[:0]
	payloadPool.Put(p)
}
