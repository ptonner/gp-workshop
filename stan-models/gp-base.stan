data {
  int<lower=1> N;
  vector[N] y;
  real x[N];
}
parameters {
  //constrain hyperparameters to be positive
  real<lower=0> length_scale;
  real<lower=0> alpha; // RBF variance
  real<lower=0> sigma; //observation noise
  vector[N] f_eta;
}
transformed parameters {
  vector[N] f;
  {
    // define unneeded variables inside brackets
    // to prevent storage by sampler
    matrix[N, N] L_cov;
    matrix[N, N] cov;
    cov = cov_exp_quad(x, alpha, length_scale);
    for (n in 1:N)
      cov[n, n] = cov[n, n] + 1e-12;
    L_cov = cholesky_decompose(cov);
    f = L_cov * f_eta;
  }
}
model {
  length_scale ~ gamma(2, 2);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
  f_eta ~ normal(0, 1);
  y ~ normal(f, sigma);
}
