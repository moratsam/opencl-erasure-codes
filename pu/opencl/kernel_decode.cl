kernel void decode(global uchar *exp_table, global uchar *log_table, global uchar *inv, global uchar *data, uchar n, global uchar *output) {
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
	uchar w[3]; // tfw no idea why this werks when n>3.
	for(int i=0; i<n; i++){
		w[i] = 0;
	}
	for (int r=0; r<n; r++) { //for every row in inv
		for (int j=0; j<=r; j++) {
			if (r == j) { //diagonal values were overwritten in LU, but pretend they're still 1
				w[r] = add(w[r], data[gid1+j*max_gid1]);
			} else {
				w[r] = add(w[r], mul(inv[n*r+j], data[gid1+j*max_gid1], exp_table, log_table));
			}
		}
	}

	/*
	if (gid1==1){
		printf("w: [");
		for(int i=0; i<n; i++) {
			printf("%d ", w[i]);
		}
		printf("]\n");
	}
	*/

	for (int r=n-1; r>=0; r--) {
		for (int j=n-1; j>=r; j--) {
			output[r+n*gid1] = add(output[r+n*gid1], mul(inv[j+n*r], w[j], exp_table, log_table));
		}
	}

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


