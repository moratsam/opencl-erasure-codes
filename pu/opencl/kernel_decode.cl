kernel void decode(global uchar *exp_table, global uchar *log_table, global uchar *inv, global uchar *data, uchar n, global uchar *output) {
	int z = get_global_id(1);
	int max_z = get_global_size(1);
	/*
	if (z==1) {
		printf("\nlol: %d\n", n);
		printf("lmao: %d\n", inv[n+1]);
		printf("kek: %d\n", data[n*z]);
		printf("max z: %d\n\n", max_z);
	}

	if (z==1){
		printf("enc word: [");
		for(int i=0; i<n; i++) {
			printf("%d ", data[z+i*max_z]);
		}
		printf("]\n");
	}
	*/

	//calculate W := (L^-1)[enc_word]
	uchar w[3];
	for(int i=0; i<n; i++){
		w[i] = 0;
	}
	for (int r=0; r<n; r++) { //for every row in inv
		for (int j=0; j<=r; j++) {
			if (r == j) { //diagonal values were overwritten in LU, but pretend they're still 1
				w[r] = add(w[r], data[z+j*max_z]);
			} else {
				w[r] = add(w[r], mul(inv[n*r+j], data[z+j*max_z], exp_table, log_table));
			}
		}
	}

	/*
	if (z==1){
		printf("w: [");
		for(int i=0; i<n; i++) {
			printf("%d ", w[i]);
		}
		printf("]\n");
	}
	*/

	for (int r=n-1; r>=0; r--) {
		for (int j=n-1; j>=r; j--) {
			output[r+n*z] = add(output[r+n*z], mul(inv[j+n*r], w[j], exp_table, log_table));
		}
	}

	/*
	if (z==1){
		printf("data word: [");
		for(int i=0; i<n; i++) {
			printf("%d ", output[z+i*max_z]);
		}
		printf("]\n");
	}
	*/
}


