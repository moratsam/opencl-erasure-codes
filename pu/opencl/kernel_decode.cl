kernel void decode(global uchar *exp_table, global uchar *log_table, global uchar *inv, global uchar *data, global uchar *output) {
	int lid0 = get_local_id(0);
	int lid1 = get_local_id(1);
	int gid1 = get_global_id(1);
	int max_gid1 = get_global_size(1);

	//calculate W := (L^-1)[enc_word]
	local uchar w[MAX_LID1*SIZE_N];
	uchar res = 0;
	for (int j=0; j<=lid0; j++) {
		if (lid0 == j) { //diagonal values were overwritten in LU, but pretend they're still 1
			res = add(res, data[gid1+j*max_gid1]);
		} else {
			res = add(res, mul(inv[SIZE_N*lid0+j], data[gid1+j*max_gid1], exp_table, log_table));
		}
	}
	w[SIZE_N*lid1+lid0] = res;

	barrier(CLK_LOCAL_MEM_FENCE);

	res = 0;
	for (int j=SIZE_N-1; j>=lid0; j--) {
		res = add(res, mul(inv[j+SIZE_N*lid0], w[SIZE_N*lid1+j], exp_table, log_table));
	}
	output[lid0+SIZE_N*gid1] = res;
}


