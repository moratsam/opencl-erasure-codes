package opencl

import (
	"sync"
	
	"github.com/jgillich/go-opencl/cl"
	"github.com/moratsam/etherscan/pipeline"
)

var payloadPool = sync.Pool{ New: func() interface{} {return new(streamerPayload)} }

type streamerPayload struct {
	n						int // Number of encoded shards.
	n_words				int // Number of n-sized words in chunk of encoded data.
	padded_n_words		int // n_words + potential padding, so that is suits GPU size limits.
	global_work_size	[]int // GPU global work size.
	local_work_size	[]int // GPU local work size.
	host_in				[]byte // host input array of data (Is copied to the GPU).
	host_out				[]byte // host output array of data (GPU output is copied here).
	cl_buf_in			*cl.MemObject // device input array of data (GPU reads from here).
	cl_buf_out			*cl.MemObject // device output array of data (GPU writes here).
}

// Doesn't really clone, cloning isn't needed.
func (p *streamerPayload) Clone() pipeline.Payload {
	return payloadPool.Get().(*streamerPayload)
}

func (p *streamerPayload) MarkAsProcessed() {
	// Clear up resources before putting the payload struct back in the pool.
	p.global_work_size = p.global_work_size[:0]
	p.local_work_size = p.local_work_size[:0]
	p.host_in = p.host_in[:0]
	p.host_out = p.host_in[:0]
	p.cl_buf_in.Release()
	p.cl_buf_out.Release()
	payloadPool.Put(p)
}
