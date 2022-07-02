kernel void decode(global uchar *exp_table, global uchar *log_table, global uchar *inv, global uchar *data, uchar n, global uchar *output) {
	int lid0 = get_local_id(0);
	int lid1 = get_local_id(1);
	int gid1 = get_global_id(1);
	int max_gid1 = get_global_size(1);
	/*
	if (gid1==1) {
		printf("\nlol: %d\n", n);
		printf("lmao: %d\n", inv[n+1]);
		printf("kek: %d\n", data[n*gid1]);
		printf("max gid1: %d\n\n", max_gid1);
	}

	if (gid1==1){
		printf("enc word: [");
		for(int i=0; i<n; i++) {
			printf("%d ", data[gid1+i*max_gid1]);
		}
		printf("]\n");
	}
	*/

	//calculate W := (L^-1)[enc_word]
	local uchar w[32*4]; // TODO pass n as compile arg instead of hardcoding n=4.
	uchar res = 0;
	for (int j=0; j<=lid0; j++) {
		if (lid0 == j) { //diagonal values were overwritten in LU, but pretend they're still 1
			res = add(res, data[gid1+j*max_gid1]);
		} else {
			res = add(res, mul(inv[n*lid0+j], data[gid1+j*max_gid1], exp_table, log_table));
		}
	}
	w[n*lid1+lid0] = res;

	barrier(CLK_LOCAL_MEM_FENCE);

	/*
	if (gid1==1){
		printf("w: [");
		for(int i=0; i<n; i++) {
			printf("%d ", w[i]);
		}
		printf("]\n");
	}
	*/

	res = 0;
	for (int j=n-1; j>=lid0; j--) {
		res = add(res, mul(inv[j+n*lid0], w[n*lid1+j], exp_table, log_table));
	}
	output[lid0+n*gid1] = res;

	/*
	if (gid1==1){
		printf("data word: [");
		for(int i=0; i<n; i++) {
			printf("%d ", output[gid1+i*max_gid1]);
		}
		printf("]\n");
	}
	*/
}


