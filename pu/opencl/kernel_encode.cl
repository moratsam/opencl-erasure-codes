kernel void encode(global uchar *exp_table, global uchar *log_table, global uchar *mat, global uchar *data, global uchar *output) {
	int gid1 = get_global_id(1);
	int lid0 = get_local_id(0);
	int max_gid1 = get_global_size(1);
	int max_lid0 = get_local_size(0);
	/*
	if (gid0 == 1 && gid1==1) {
		printf("lol: %d\printf("kkk: %d\7", res);
	}
	*/
}

