package utils

//create cauchy matrix of dimensions (n+k)xn
//every n rows suffice to reconstruct the data
func CreateCauchy(k, n byte) [][]byte{
	mat := make([][]byte, n+k)
	for i := range mat {
		mat[i] = make([]byte, n)
	}

	var i, j byte
	for i=0; i<n+k; i++ {
		for j=n+k; j<2*n+k; j++ {
			mat[i][j-n-k] = Div(1, Add(i, j))
		}
	}
	return mat
}

//create an inverse of the cauchy submatrix corresponding to row indexes in row_indexes.
func CreateInverse(mat [][]byte, row_indexes []int) [][]byte {
	cauchy := createCauchySubmatrix(mat, row_indexes)
	getLU(cauchy)
	return invertLU(cauchy)
}

//-----------------------------------------

//from the cauchy matrix mat, select only rows from row_indexes
func createCauchySubmatrix(mat [][]byte, row_indexes []int) [][]byte {
	n := len(mat[0])
	submat := make([][]byte, n) //create matrix for cauchy
	for i := range submat{
		submat[i] = make([]byte, n)
	}

	for i := range submat { //populate it with rows from whole cauchy matrix
		submat[i] = mat[row_indexes[i]][:]
	}

	return submat
}



func getLU(mat [][]byte) {
	dim := byte(len( mat[0] ))

	var i, row_ix, col_ix byte
	for i=0; i<dim; i++{
		if mat[i][i] == 0{
			continue
		}
		for row_ix=i+1; row_ix<dim; row_ix++{
			//derive factor to destroy first elemnt
			mat[row_ix][i] = Div(mat[row_ix][i], mat[i][i])
			//subtract (row i's element * factor) from every other element in row
			for col_ix=i+1; col_ix<dim; col_ix++{
				mat[row_ix][col_ix] = Sub(mat[row_ix][col_ix], Mul(mat[i][col_ix],mat[row_ix][i]))
			}
		}
	}
}

func invertLU(mat [][]byte) [][]byte {
	dim := len( mat[0] )

	side := make([][]byte, dim) //create side identity matrix
	for i := range side {
		side[i] = make([]byte, dim)
		side[i][i] = 1
	}

	//invert U by adding an identity to its side. When U becomes identity, side is inverted U.
	//no operations on U actually need to be performed, just their effects on the side
	//matrix are being recorded.
	var i, j, k int
	for i=dim-1; i>=0; i-- { //for every row
		for j=dim-1; j>i; j-- { //for every column
			for k=dim-1; k>=j; k-- { //subtract row to get a 0 in U, reflect this change in side
				side[i][k] = Sub(side[i][k], Mul(mat[i][j], side[j][k]))
			}
		}
		if mat[i][i] == 0{
			continue
		} else {
			//Divide mat[i][i] by itself to get a 1, reflect this change in whole line of side
			for j=dim-1; j>=0; j-- {
				side[i][j] = Div(side[i][j], mat[i][i])
			}
		}
	}

	//get inverse of L
	for i=0; i<dim; i++ {
		for j=0; j<i; j++ {
			for k=0; k<=j; k++ {
				//since an in-place algo was used for LU decomposition,
				//diagonal values of LU were overwritten by U,
				//whereas L expects them to be equal to 1
				//in this case, no Mul should be performed (to siMulate Multiplying by 1)
				if j == k { 
					side[i][k] = Sub(side[i][k], mat[i][j])
				} else {
					side[i][k] = Sub(side[i][k], Mul(mat[i][j], side[j][k]))
				}
			}
		}
	}

	//inverse matrix is now the side matrix! because m.inv kinda became identity matrix
	//kinda, because no changes to m.inv were actually recorded
	return side
}

