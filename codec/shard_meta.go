package codec

import (
	"os"
	"sort"

	"github.com/moratsam/opencl-erasure-codes/io"
	u "github.com/moratsam/opencl-erasure-codes/utils"
)

type shardMeta struct {
	ix 			byte // Index of the cauchy row used to encode this shard.
	n 				byte // Value of parameter n used for encoding.
	padding		byte // Padding used for encoding (in case file size % n was not 0).
	cauchy_row	[]byte // Cauchy row used to encode this shard.
}

func (s shardMeta) marshal() []byte {
	data := []byte{s.ix, s.n, s.padding}
	return append(data, s.cauchy_row...)
}

func unmarshal(data []byte) shardMeta{
	return shardMeta {
		ix				: data[0],
		n				: data[1],
		padding		: data[2],
		cauchy_row	: data[3:],
	}
}

func readShardMeta(f *os.File) (shardMeta, error) {
	data, err := io.ReadFrom(f, 3)	
	if err != nil {
		return shardMeta{}, u.WrapErr("read shard meta", err)
	}

	cauchy_row, err := io.ReadFrom(f, int64(data[1]))
	if err != nil {
		return shardMeta{}, u.WrapErr("read cauchy row", err)
	}

	data = append(data, cauchy_row...)
	return unmarshal(data), nil
}

// Write shard metadata to each shard.
func metaToShards(shards []*os.File, n, padding byte, mat [][]byte) error {
	var err error
	for i,shard := range shards {
		sm := shardMeta{byte(i), n, padding, mat[i]}
		err = io.WriteTo(shard, sm.marshal())
		if err != nil {
			return u.WrapErr("meta to shards", err)
		}
	}
	return nil
}

func toShards(shards []*os.File, data [][]byte) error {
	for i,shard := range shards {
		if err := io.WriteTo(shard, data[i]); err != nil {
			return u.WrapErr("to shards", err)
		}
	}
	return nil
}

// Each shard contains an index of it's cauchy row.
// The cauchy rows and the shards need to be sorted in increasing order based on row indices.
// Sorts the input array based on row indices.
// Returns values for n and padding and the inverse cauchy matrix.
func metaFromShards(shards []*os.File) (byte, byte, [][]byte, error){
	row_ixs := make([]int, 0, len(shards)) // Contains the cauchy_row indices of shards.
	row_to_shard_ix := make(map[int]int)	// [row_ix] -> index of shard in shards input array.
	shard_metas := make([]shardMeta, 0, len(shards)) // shardMetas for shards in input array.

	// Get shard metas, fill up the maps and arrs which will be used to sort the Shards.
	for i,f := range shards {
		sm, err := readShardMeta(f)
		if err != nil {
			return 0, 0, nil, err
		}
		row_to_shard_ix[int(sm.ix)] = i
		shard_metas = append(shard_metas, sm)
		row_ixs = append(row_ixs, int(sm.ix))
	}

	// Sort the row indices.
	sort.Ints(row_ixs)

	n := shard_metas[0].n					// Is the same for all shards.
	padding := shard_metas[0].padding	// Is the same for all shards.

	// Construct the cauchy submatrix by stacking cauchy_rows from sorted shard_metas.
	mat := make([][]byte, n)
	for i := range mat {	
		shard_ix := row_to_shard_ix[row_ixs[i]]
		mat[i] = shard_metas[shard_ix].cauchy_row
	}

	// Invert the cauchy submatrix.
	inv := u.Invert(mat)

	// Sort input array of shards.
	sorted_shards := make([]*os.File, n)
	for i := range shards {
		shard_ix := row_to_shard_ix[row_ixs[i]]
		sorted_shards[i] = shards[shard_ix]
	}
	shards = sorted_shards

	return n, padding, inv, nil
}

