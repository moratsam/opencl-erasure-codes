kernel void encode(global uchar *exp_table, global uchar *log_table, global uchar *mat, global uchar *data, global uchar *output) {
	int gid1 = get_global_id(1);
	int lid0 = get_local_id(0);
	int max_gid1 = get_global_size(1);
	/*
	if (gid0 == 1 && gid1==1) {
		printf("lol: %d\n", n);
		printf("lmao: %d\n", mat[n*gid0]);
		printf("kek: %d\n", data[n*gid1]);
		printf("max gid1: %d\n", max_gid1);
	}
	*/

	uchar res = 0;
	for (int c=0; c<SIZE_N; c++){
		res = add(res, mul(mat[SIZE_N*lid0 + c], data[SIZE_N*gid1+c], exp_table, log_table));
	}

	output[lid0*max_gid1 + gid1] = res;

	/*
	if (gid0==0 && gid1==1) {
		printf("kkk: %d\n", res);
	}
	*/
}

