---
title: "Practical: Poisson regression in health"
subtitle: "MEDISS 2025"
authors:
  - name: Kinh Nguyen, PhD
    email: kinh.nguyen@uni-heidelberg.de
    affiliations:
      - ref: high
affiliations:
  - id: high
    name: Heidelberg Institute for Global Health, <br>Heidelberg University
execute: 
  cache: true
  echo: true
  warning: false
editor:
  render-on-save: true
date: today
date-format: long
format: 
  revealjs:
    theme: karto.scss
    slide-number: c
    toc: true
    toc-depth: 2
footer: "--- MEDISS 2025 ---"
toc: false
toc-depth: 1
menu: false
chalkboard: true
license: "CC BY-SA"
number-sections: true
number-depth: 1
slide-level: 2
#title-slide-attributes:
  # data-background-image: ./people-intro.svg
  # data-background-opacity: ".1"
  # data-background-color: "#F2F2F2"
  # data-background-size: cover
  # data-background-position: bottom left
transition: fade
callout-appearance: minimal
bibliography: /Users/knguyen/zotero.bib
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, message = FALSE, warning = FALSE, cache = TRUE)
```

# Poisson regression recap

- Discrete distribution, probability of a given number of events occurring in a fixed interval of time or space (count data)
  $$P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$
  * $P(k; \lambda)$ is the probability of observing $k$ events.
  * $k$ is the number of occurrences (an integer, $k = 0, 1, 2, ...$).
  * $\lambda > 0$: average number of events per interval (mean).
  * Variance: $\text{Var}[X] = \lambda$.

* Number of animal aggression over space and time in Mexico [@nguyen2023]
* Number of health care visits per duration across time and countries in Europe [@nguyen2024b]

## Capacity planning for an ICU center

An ICU receives an average of **5 accidents per hour** ($\lambda = 5$). What is the probability of receiving exactly **3 calls** in a given hour ($k = 3$).
$$
P(3; 5) = \frac{5^3 e^{-5}}{3!} \approx 0.1404
$$

::: {.callout-note}
**14.04%** chance of the ICU receiving 3 patients in any given hour.
:::

# Poisson Generalized Linear Model

The response variable, $Y_i$, for observation $i$ follows a Poisson distribution with a mean $\mu_i$.
    $$ Y_i \sim \text{Poisson}(\mu_i) $$
The model's predictors are linear combination of covariates
    $$ \eta_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_p x_{pi} $$
The **log link** function connects the mean to the linear predictor (positive mean)
    $$ \log(\mu_i) = \eta_i $$

## Simulate a Poisson data in R

We simulated data and fit a simple Poisson model, including

- intercept and one continuous covariate's effect
- exposure variable (e.g., person-days of observation)

```{r brms_poisson_prep}
library(brms)
library(ggplot2)

set.seed(123)
n <- 100

exposure <- sample(10:100, n, replace = TRUE) 
x <- rnorm(n)      # A continuous predictor
beta0 <- 0.5       # True intercept on the log-rate scale
beta1 <- -0.8      # True effect of x on the log-rate

# Calculate the mean count (mu). The linear predictor now models the RATE.
# mu = exposure * rate
# log(mu) = log(exposure) + log(rate)
# log(rate) = beta0 + beta1 * x
mu <- exposure * exp(beta0 + beta1 * x) 

y <- rpois(n, lambda = mu)   # Generate counts based on the rate and exposure
sim_data <- data.frame(y, x, exposure)
```

## The count/rate data 

```{r show_count}
#| fig-cap: Illustration of rate distribution
#| fig-width: 4
#| fig-height: 3
sim_data$rate = sim_data$y / sim_data$exposure
ggplot(sim_data, aes(rate)) + geom_histogram()

head(sim_data, 3)
```

## Fitting/checking a simple Poisson regression

- Fit the model with an offset
- The offset term must be on the log scale (Why?).

```{r brms_poisson}
fit_brms <- brm(
  y ~ x + offset(log(exposure)),
  data = sim_data,
  family = poisson(link = "log"),
  prior = c(
    prior(normal(0, 2), class = "b"),
    prior(normal(0, 5), class = "Intercept")
  ),
  chains = 4, iter = 2000, warmup = 1000, cores = 4,
  seed = 123
)
```

---

## Results {.unlisted}

```{r summary}
summary(fit_brms)
```

::: {.callout-note}
* `Intercept`: the expected **log count** when predictor `x` is zero.
* `x`: for one-unit increase in `x`, the **log count** increases by 1.18. 
  * `exp(x)` is the **rate ratio (RR)**
  * one-unit increase in `x` increases the count by RR **times**.
* `l-95% CI, u-95% CI`: the 95% credible interval 
  * evidence that `x` is associated with `y`?
:::

## Posterior Predictive Check {.unlisted}

How well our model simulates data that looks like observed data.

```{r pp_check_poisson}
#| fig-cap: Distribution of actual data as the dark histogram with several simulated datasets from the fitted model's posterior distribution (blue).
pp_check(fit_brms, type = "hist", ndraws = 11)
```

# Overdispersion

- Assumption that the **mean equals the variance** 
- Variance > mean, **overdispersion**.
- Negative Binomial model: 
  - add **dispersion parameter** to model extra variance.

$$ Y_i \sim \text{Negative-Binomial}(\mu_i, \phi) $$
$$ \text{Var}(Y_i) = \mu_i + \frac{\mu_i^2}{\phi} $$

- $\phi \to \infty$, $\text{Var}(Y_i) \to \mu_i$, Negative Binomial $\to$ Poisson.

## Simulate and fit a NB and model comparison

 Simulate overdispersed data and fit a Negative Binomial model to compare its performance against the original Poisson model.

```{r nb_model}
# 1. Simulate Overdispersed Data with an Exposure Variable
set.seed(456)
n <- 100
exposure <- sample(10:100, n, replace = TRUE)
x <- rnorm(n)
beta0 <- 0.5   # True intercept on the log-rate scale
beta1 <- -0.8  # True effect of x on the log-rate
phi <- 2       # A smaller phi introduces more overdispersion

# Calculate mu, incorporating the exposure
mu_overdispersed <- exposure * exp(beta0 + beta1 * x)
y_overdispersed <- rnbinom(n, size = phi, mu = mu_overdispersed)
overdispersed_data <- data.frame(y = y_overdispersed, x, exposure)
```

```{r plot_nb}
#| fig-cap: Simulated overdispersed data
ggplot(overdispersed_data, aes(y = y/exposure)) + geom_histogram()
```

## Fit Negative Binomial model with offset {.unlisted}

```{r fit_nb}
fit_nb_brms <- brm(
  y ~ x + offset(log(exposure)),
  data = overdispersed_data,
  family = negbinomial(link = "log"),
  prior = c(
    prior(normal(0, 2), class = "b"),
    prior(normal(0, 5), class = "Intercept"),
    prior(gamma(0.01, 0.01), class = "shape") # Prior for phi
  ),
  chains = 4, iter = 2000, warmup = 1000, cores = 4,
  seed = 456
)

# 3. Fit a Poisson model to the same data for comparison
fit_poisson_overdisp <- brm(
  y ~ x + offset(log(exposure)),
  data = overdispersed_data,
  family = poisson(link = "log"),
  chains = 4, iter = 2000, warmup = 1000, cores = 4,
  seed = 456
)
```

## Model comparison {.unlisted}

Compare models using LOO-CV (leave one out cross validation)

```{r loo}
loo_nb <- loo(fit_nb_brms)
loo_poisson <- loo(fit_poisson_overdisp)

loo_compare(loo_nb, loo_poisson)
```

- ranks the models by their expected log predictive density. 
- The model with the highest value is preferred 
- test for difference: `se_diff`

# Spatial regression for disease counts

- data points that are close in space are often more similar than those far apart.
- extend the Poisson model to include **spatial random effects** and an **offset** for the population at risk.

$$ Y_i \sim \text{Poisson}(\mu_i) $$
$$ \log(\mu_i) = \log(E_i) + \beta_0 + u_i $$

  * $Y_i$ is the observed disease count in area $i$.
  * $\log(E_i)$ is the **offset** (e.g., population size): model the **relative risk**.
  * $\beta_0$ the average log relative risk across all areas.
  * $u_i$: 
    * spatial autocorrelation **Intrinsic Conditional Autoregressive (ICAR)** 
    * effect in one area is related to the average of its neighbors.

## Mapping Scottish lip cancer risk

- Lip cancer counts from 56 counties in Scotland.
- model SIDS cases in 1974 (SID74) 
  - using births in 1974 (BIR74) as the exposure/offset

```{r spatial_load_data}
#| label: fig-sids-map
#| fig-cap: "Map of North Carolina counties"
#| layout-ncol: 2
#| column-widths: [40, 60] 
library(spdep)
sf_use_s2(TRUE)
library(spData)

library(spdep)
nc.sids <- st_read(system.file("shapes/sids.gpkg", package="spData")[1], quiet=TRUE)
row.names(nc.sids) <- as.character(nc.sids$FIPSNO)

ggplot(nc.sids) + 
  geom_sf(aes(fill = FIPSNO)) + 
  geom_sf_text(aes(label = FIPSNO)) + 
  theme_void() +
  guides(fill = 'none')
```

## Fit a simple spatial model

Creating neighboring structure to inform the model

::: .columns
::: {.column width="40%"}
```{r image_nb}
nc.sids$id <- 1:nrow(nc.sids)
adj_matrix <- poly2nb(nc.sids)
W <- nb2mat(adj_matrix, style = "B", zero.policy = TRUE)
rownames(W) <- colnames(W) <- nc.sids$id
```
:::

::: {.column width="60%"}
```{r}
#| label: fig-nbmt
#| fig-cap: Neighboring matrix
par(pty = "s")
image(W)
```

:::
:::

## Fit the spatial (ICAR) model {.unlisted}

- Fit the model with `icar` and exposure
- Extract the spatial effect 
- Convert to relative risk (`exp`)
 
```{r fit_spatial}
fit_spatial_sids <- brm(
  SID74 | rate(BIR74) ~ 1 + car(W, gr = id, type = 'icar'),
  data = nc.sids,
  data2 = list(W = W),
  family = poisson(link = "log"),
  chains = 4, iter = 4000, warmup = 2000, cores = 4,
  control = list(adapt_delta = 0.95),
  seed = 555
)
cov_effects <- posterior_summary(fit_spatial_sids)
spatial_effects <- posterior_summary(fit_spatial_sids, variable = "rcar")
nc.sids$relative_risk <- exp(spatial_effects[,"Estimate"])
```

## The relative risk map

```{r}
#| label: fig-rrmap
#| fig-cap: "Relative Risk of SIDS in North Carolina (1974)"
ggplot(nc.sids, aes(fill = relative_risk)) +
  geom_sf(color = "black", size = 0.2) +
  scale_fill_viridis_c(name = "Relative Risk") +
  theme_void() +
  labs(caption = "Data source: spData package")
```

# Read more {.unlisted}

- Disease mapping/cluster analysis [@olsenClusterAnalysisDisease1996]
- Bayesian regression [@gelman2013]
- Bayesian spatio-temporal model[@knorr-heldBayesianModellingInseparable2000]
