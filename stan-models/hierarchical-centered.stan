data {
    int<lower=0> N; // number of observations
    int<lower=1> G; // number of groups
    int<lower=1, upper=G> group[N]; // group assignment
    real<lower=0> sigma; // observation noise
    real y[N]; // observations
}
parameters {
    real mu; // grand mean
    real<lower=0> tau; // group variance
    real theta[G]; // group mean
}
transformed parameters{
    real yhat[N];

    for(i in 1:N)
        yhat[i] = theta[group[i]];
}
model {
    mu ~ normal(0, 5);
    tau ~ cauchy(0, 2.5);
    theta ~ normal(mu, tau);
    y ~ normal(yhat, sigma);
}
