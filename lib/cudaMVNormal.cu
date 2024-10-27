#include <assert.h>

#include "cudaMVNormal.hu"

/*
 * Computes \sum_{i}^{N} x_i y_i for x, y \in \mathbb{R}^{N}.
 */
__device__ double devVecDot(const size_t N, const double* x, const double* y) {
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);
	// x == y allowed

	double sum = 0;
	for(size_t i = 0; i < N; ++i) {
		sum += x[i] * y[i];
	}
	return sum;
}

/*
 * Computes z_{i} \gets x_{i} - y_{i} for x, y \in \mathbb{R}^N.
 */
__device__ void devVecMinus(const size_t N, double* z, const double* x, const double* y) {
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);
	// x == y allowed

	for(size_t i = 0; i < N; ++i) {
		z[i] = x[i] - y[i];
	}
}

/*
 * Solves the lower triangular system L^T x = b for x, b \in \mathbb{R}^{N}, 
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 * @details 
 */
/**
 * @brief Solves the lower triangular system L * x = b for x using forward substitution.
 *
 * @param N The size of the matrix and vectors.
 * @param L Pointer to the lower triangular matrix L stored in row-major order.
 * @param x Pointer to the solution vector x. Can be the same as b.
 * @param b Pointer to the right-hand side vector b.
 *
 * This function assumes that L is a non-singular lower triangular matrix.
 * It performs forward substitution to solve for x in the equation L * x = b.
 */
__device__ void devSolveLowerTri(const size_t N, const double* L, double* x, const double* b) {
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

	for(size_t i = 0; i < N; ++i) {
		double sum = 0.0;
		if(i > 0) {
			for(size_t j = 0; j <= i - 1; ++j) {
				sum += L[i * N + j] * x[j];
			}
		}

		x[i] = (b[i] - sum) / L[i * N + i];
	}
}

/*
 * Solves the upper triangular system L^T x = b for x, b \in \mathbb{R}^{N}, 
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 */
__device__ void devSolveLowerTriT(const size_t N, const double* L, double* x, const double* b) {
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

	// treat L as an upper triangular matrix U
	for(size_t i = 0; i < N; i++) {
		size_t ip = N - 1 - i;
		double sum = 0;
		for(size_t j = ip + 1; j < N; ++j) {
			sum += L[j * N + ip] * x[j];
		}

		x[ip] = (b[ip] - sum) / L[ip * N + ip];
	}
}


/*
 *
 */
__device__ double devLogMVNormNormalizer(
	const size_t pointDim,
	const double* sigmaL
) {
	double det = 1.0;
	for(size_t i = 0; i < pointDim; ++i) {
		det *= sigmaL[i * pointDim + i]; // ad
	}
	det *= det; // ad * ad = det(A)^2

	return -0.5 * log( 2.0 * M_PI ) * pointDim - 0.5 * log(det); // equation 3
}

/*
 * Computes log( p(x | mu, Sigma ) ) for multivariate normal distribution with 
 * parameters mu (mean), and Sigma (covariance).
 */
__device__ double devLogMVNormDist(
	const size_t pointDim,
	const double* x, const double* mu, const double* sigmaL,
	double* u, double* v
) {
	devVecMinus(pointDim, v, x, mu); // v <- x - mu
	devSolveLowerTri(pointDim, sigmaL, u, v); // u <- u s.t. L u = (x - mu), slove u from L * u = v = x - mu
	devSolveLowerTriT(pointDim, sigmaL, u, u); // u <- v s.t. L^T v = u
	return devLogMVNormNormalizer(pointDim, sigmaL) - 0.5 * devVecDot(pointDim, u, v); // equation 3
}

/**
 * @brief Launch number of data points threads
 * @details update the logProb array with the log probability of each data point, eq3
 */
__global__ void kernLogMVNormDist(
	const size_t numPoints, const size_t pointDim, 
	const double* X, double* mu, double* sigmaL,
	double* logProb
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	// redundant for 1024 dimensions
	double u[1024];
	double v[1024];
	
	// equation 3
	logProb[i] = devLogMVNormDist(
		pointDim, 
		& X[i * pointDim], mu, sigmaL,
		u, v
	);
}

