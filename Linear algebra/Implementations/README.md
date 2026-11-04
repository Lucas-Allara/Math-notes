# Implementations

The algorithms of the computational sections of the [Linear algebra notes](<../Linear algebra notes.pdf>) in Python, with matrices as lists of rows and support for `Fraction`.

```
Implementations/
├── block_product.py           direct and recursive 2×2 block multiplication (sections 10.1–10.2)
├── lup.py                     PA = LU with partial pivoting, substitutions, solve, determinant and inverse (10.3)
├── qr_gram_schmidt.py         A = QR by classical Gram-Schmidt (10.4)
├── least_squares.py           normal equations and the QR route, R x̂ = Qᵀb (10.4)
├── orthogonal_projection.py   projection onto Col(A) as Q Qᵀ b (sections 2 and 10.4)
├── power_method.py            dominant eigenvalue with the Rayleigh quotient, and deflation for symmetric matrices (10.5)
├── pca.py                     principal components from the covariance matrix and the power method (sections 9 and 10.5)
├── kmeans_lloyd.py            Lloyd's procedure with the cost history (11.1)
└── gradient_descent.py        fixed step and the average of the iterates, the variant of the convergence theorem (11.2)
```
