kernel void encode(global uchar *exp_table, global uchar *log_table, global uchar *mat, global uchar *data, uchar n, global uchar *output) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int max_j = get_global_size(1);
	/*
	if (i == 1 && j==1) {
		printf("lol: %d\n", n);
		printf("lmao: %d\n", mat[n*i]);
		printf("kek: %d\n", data[n*j]);
		printf("max j: %d\n", max_j);
	}
	*/

	uchar res = 0;
	for (int c=0; c<n; c++){
		res = add(res, mul(mat[n*i + c], data[n*j+c], exp_table, log_table));
	}

	output[i*max_j+j] = res;

	/*
	if (i==0 && j==1) {
		printf("kkk: %d\n", res);
	}
	*/
}

