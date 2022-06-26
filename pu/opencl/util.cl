uchar add(uchar a, uchar b) {
	return a^b;
}

uchar sub(uchar a, uchar b) {
	return a^b;
}

uchar div(uchar a, uchar b, uchar *exp_table, uchar *log_table) {
	if (a == 0) {
		return 0;
	} else {
		return exp_table[log_table[a] + (255 - log_table[b])];
	}
}

uchar mul(uchar a, uchar b, uchar *exp_table, uchar *log_table) {
	if (a==0 || b==0) {
		return 0;
	}
	return exp_table[log_table[a] + log_table[b]];
}
