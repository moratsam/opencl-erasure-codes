package util

// Here basic operations on polynomials over the galois field 2^8 are implemented.

var prime = 0x11d
var exp_table = make([]byte, 512)
var log_table = make([]byte, 256)
var tables_initialised = false

func GetTables() ([]byte, []byte) {
	if ! tables_initialised {
		init_tables()
	}
	return exp_table, log_table	
}

func Add(a, b byte) byte {
	return a^b
}

func Sub(a, b byte) byte {
	return a^b
}

func Mul(a, b byte) byte {
	if ! tables_initialised {
		init_tables()
		tables_initialised = true
	}
	if a==0 || b==0 {
		return 0
	}
	return exp_table[int(log_table[a]) + int(log_table[b])]
}

func Div(a, b byte) byte {
	if ! tables_initialised {
		init_tables()
		tables_initialised = true
	}
	if a == 0{
		return 0
	} else if b == 0{
		panic("division by zero")
	} else{
		return exp_table[int(log_table[a]) + 255 - int(log_table[b])]
	}
}

//-----------------------------------------

//use generator 2 to init log and exp tables
func init_tables() {
	x := byte(1)
	for i:=0; i<255; i++ {
		exp_table[i] = x
		log_table[x] = byte(i)
		x = mul_costly(x, 2)
	}
	for i:=255; i<512; i++ {
		exp_table[i] = exp_table[i-255]
	}
}

// Calculate bit length.
func length(a int) int {
	result := 0
	for i:=0; a>>i >0; i++ {
		result++
	}
	return result
}

func mul_costly(a, b byte) byte {
	result := 0
	for i:=0; a>>i > 0; i++ { //iterate over the bits of a
		if (a & (1<<i)) > 0 { // if current bit is 1
			result ^= int(b)<<i //xor b multiplied by this power of 2 to result
		}
	}

	len1, len2 := length(result), length(prime)
	if len1 < len2 {
		return byte(result)
	}
	for i:=len1-len2; i > -1; i-- { //while result is not smaller than the prime
		if (result & (1<<(i+len2-1))) > 0{ //if current bit is 1
			result ^= prime << i //align divisor with the result and subtract its value
		}
	}
	return byte(result)
}


