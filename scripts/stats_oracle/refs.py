"""References for the stats oracle (scripts/stats_oracle/README.md).

`scipy_ref(row)` gives what scipy / statsmodels return for a case row (the
libraries a user would otherwise call); `mp_ref(row, output, arg)` gives the
mpmath value of the same quantity from its closed form (or an exact sum),
evaluated at the current `mp.prec`, for an exact argument.  The comparator
evaluates `mp_ref` at two precisions and trusts only digits on which they
agree.  Values: an `mpf`/`mpc`, a `Fraction` (exact), or one of the strings
"inf", "-inf", "nan" (the quantity is that), or `None` (no reference).
"""
import math
import sys
from fractions import Fraction as F

import mpmath as mp

# Exact rationals with thousands of digits are compared as numbers here.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

DISC = {"bernoulli", "binomial", "poisson", "geometric", "negative_binomial", "hypergeometric",
        "discrete_uniform", "finite"}


def frac(s):
    s = s.strip()
    if "/" in s:
        n, d = s.split("/")
        return F(int(n), int(d))
    return F(int(s))


def q_of(v):
    """An exact rational as `N` or `N/D` (the case-file notation)."""
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def params_of(row):
    if row.target == "finite":
        out = []
        for item in row.params.split(","):
            v, p = item.split(":")
            out.append((frac(v), frac(p)))
        return out
    return [frac(s) for s in row.params.split(",")]


def M(v):
    """An exact rational (or int/float) as an mpf at the current precision."""
    if isinstance(v, F):
        return mp.mpf(v.numerator) / v.denominator
    return mp.mpf(v)


def fields(s):
    return dict(kv.split("=", 1) for kv in s.split(";") if kv)


def qlist(s):
    return [frac(v) for v in s.split(",")] if s else []


# ───────────────────────────── scipy ─────────────────────────────

def scipy_dist(fam, P):
    from scipy import stats
    f = [float(v) for v in P] if fam != "finite" else None
    if fam == "normal": return stats.norm(loc=f[0], scale=f[1])
    if fam == "uniform": return stats.uniform(loc=f[0], scale=float(P[1] - P[0]))
    if fam == "exponential": return stats.expon(scale=float(1 / P[0]))
    if fam == "gamma": return stats.gamma(f[0], scale=f[1])
    if fam == "chi_squared": return stats.chi2(f[0])
    if fam == "beta": return stats.beta(f[0], f[1])
    if fam == "cauchy": return stats.cauchy(loc=f[0], scale=f[1])
    if fam == "laplace": return stats.laplace(loc=f[0], scale=f[1])
    if fam == "logistic": return stats.logistic(loc=f[0], scale=f[1])
    if fam == "log_normal": return stats.lognorm(f[1], scale=math.exp(f[0]))
    if fam == "student_t": return stats.t(f[0])
    if fam == "f": return stats.f(f[0], f[1])
    if fam == "weibull": return stats.weibull_min(f[1], scale=f[0])
    if fam == "pareto": return stats.pareto(f[1], scale=f[0])
    if fam == "triangular":
        a, b, c = P
        return stats.triang(float((c - a) / (b - a)), loc=f[0], scale=float(b - a))
    if fam == "bernoulli": return stats.bernoulli(f[0])
    if fam == "binomial": return stats.binom(int(P[0]), f[1])
    if fam == "poisson": return stats.poisson(f[0])
    if fam == "geometric": return stats.geom(f[0])
    if fam == "negative_binomial": return stats.nbinom(f[0], f[1])
    if fam == "hypergeometric": return stats.hypergeom(int(P[0]), int(P[1]), int(P[2]))
    if fam == "discrete_uniform": return stats.randint(int(P[0]), int(P[1]) + 1)
    if fam == "finite":
        vals = [float(v) for v, _ in P]
        probs = [float(p) for _, p in P]
        if all(v == int(v) for v in vals):
            return stats.rv_discrete(values=([int(v) for v in vals], probs))
        return None
    raise KeyError(fam)


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def scipy_dist_ref(row, output):
    """scipy's value for a `dist` row: a float, a complex, or None (no reference)."""
    P = params_of(row)
    if row.method in ("mgf", "cf"):
        return None
    d = scipy_dist(row.target, P)
    if d is None:
        return None
    m, disc = row.method, row.target in DISC
    if m in ("pdf", "cdf", "sf", "quantile"):
        a = float(frac(row.arg))
        if m == "pdf": return _float(d.pmf(a) if disc else d.pdf(a))
        if m == "cdf": return _float(d.cdf(a))
        if m == "sf": return _float(d.sf(a))
        p = frac(row.arg)
        return _float(d.isf(float(1 - p)) if p > F(1, 2) else d.ppf(float(p)))
    if m == "quantile_f64": return _float(d.ppf(float(row.arg)))
    if m == "isf_f64": return _float(d.isf(float(row.arg)))
    if m == "median": return _float(d.median())
    if m in ("mean", "variance", "skewness", "kurtosis"):
        mean, var, skew, kurt = d.stats(moments="mvsk")
        v = {"mean": mean, "variance": var, "skewness": skew, "kurtosis": kurt + 3}[m]
        return _float(v)
    if m == "entropy": return _float(d.entropy())
    return None


def scipy_kernel_ref(row):
    from scipy import special, stats
    a = [float(s) for s in row.params.split(",")]
    t = row.target
    if t == "betainc_regularized": return float(special.betainc(a[0], a[1], a[2]))
    if t == "gammainc_lower_regularized": return float(special.gammainc(a[0], a[1]))
    if t == "gammainc_upper_regularized": return float(special.gammaincc(a[0], a[1]))
    fam, fn = t.split(".")
    if fam == "norm": d, x = stats.norm(), a[0]
    elif fam == "t": d, x = stats.t(a[1]), a[0]
    elif fam == "gamma": d, x = stats.gamma(a[1], scale=a[2]), a[0]
    elif fam == "chi2": d, x = stats.chi2(a[1]), a[0]
    elif fam == "beta": d, x = stats.beta(a[1], a[2]), a[0]
    elif fam == "f": d, x = stats.f(a[1], a[2]), a[0]
    elif fam == "binom": d, x = stats.binom(a[1], a[2]), a[0]
    elif fam == "poisson": d, x = stats.poisson(a[1]), a[0]
    else: return None
    if fn == "pdf": return float(d.pdf(x))
    return float(getattr(d, fn)(x))


def _alt_sm(alt):
    return {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}[alt]


def scipy_test_ref(row):
    """{output: float} from scipy / statsmodels for a `test` row."""
    import numpy as np
    from scipy import stats
    f = fields(row.params)
    fl = lambda k: np.array([float(v) for v in qlist(f[k])])
    alt = f.get("alt", "two-sided")
    t = row.target
    if t == "t_test_one_sample":
        r = stats.ttest_1samp(fl("x"), float(frac(f["mu0"])), alternative=alt)
        return {"statistic": r.statistic, "pvalue": r.pvalue, "df": r.df}
    if t == "t_test_two_sample":
        r = stats.ttest_ind(fl("x"), fl("y"), equal_var=f["equal_var"] == "true", alternative=alt)
        return {"statistic": r.statistic, "pvalue": r.pvalue, "df": r.df}
    if t == "t_test_paired":
        r = stats.ttest_rel(fl("x"), fl("y"), alternative=alt)
        return {"statistic": r.statistic, "pvalue": r.pvalue, "df": r.df}
    if t == "binomial_test":
        r = stats.binomtest(int(f["k"]), int(f["n"]), float(frac(f["p0"])), alternative=alt)
        return {"pvalue": r.pvalue}
    if t == "fisher_exact":
        tab = [[int(c) for c in rr.split(",")] for rr in f["table"].split(":")]
        r = stats.fisher_exact(tab, alternative=alt)
        return {"statistic": r.statistic, "pvalue": r.pvalue}
    if t == "chi_square_independence":
        tab = [[float(c) for c in rr.split(",")] for rr in f["table"].split(":")]
        r = stats.chi2_contingency(tab, correction=f["correction"] == "true")
        return {"statistic": r.statistic, "pvalue": r.pvalue, "df": r.dof}
    if t == "chi_square_goodness_of_fit":
        obs = fl("observed")
        r = stats.chisquare(obs)
        return {"statistic": r.statistic, "pvalue": r.pvalue, "df": len(obs) - 1}
    if t == "z_test_proportion":
        from statsmodels.stats.proportion import proportions_ztest
        p0 = float(frac(f["p0"]))
        z, p = proportions_ztest(int(f["k"]), int(f["n"]), value=p0, prop_var=p0, alternative=_alt_sm(alt))
        return {"statistic": z, "pvalue": p}
    if t == "pearson_test":
        r = stats.pearsonr(fl("x"), fl("y"), alternative=alt)
        return {"statistic": r.statistic, "pvalue": r.pvalue}
    if t == "spearman_test":
        r = stats.spearmanr(fl("x"), fl("y"), alternative=alt)
        return {"statistic": r.statistic, "pvalue": r.pvalue}
    if t == "mann_whitney_u":
        meth = f["method"]
        r = stats.mannwhitneyu(fl("x"), fl("y"), use_continuity=meth == "asymptotic_cc", alternative=alt,
                               method="exact" if meth == "exact" else "asymptotic")
        return {"statistic": r.statistic, "pvalue": r.pvalue}
    if t == "wilcoxon_signed_rank":
        meth = f["method"]
        r = stats.wilcoxon(fl("x"), alternative=alt, correction=meth == "asymptotic_cc",
                           method="exact" if meth == "exact" else "asymptotic", zero_method="wilcox")
        return {"statistic": r.statistic, "pvalue": r.pvalue}
    if t == "kruskal_wallis":
        groups = [np.array([float(v) for v in qlist(g)]) for g in f["groups"].split(":")]
        r = stats.kruskal(*groups)
        return {"statistic": r.statistic, "pvalue": r.pvalue}
    if t == "anova_one_way":
        groups = [np.array([float(v) for v in qlist(g)]) for g in f["groups"].split(":")]
        r = stats.f_oneway(*groups)
        return {"statistic": r.statistic, "pvalue": r.pvalue}
    return {}


def model_design(row):
    f = fields(row.params)
    y = qlist(f["y"])
    cols = []
    k = 1
    while f"x{k}" in f:
        cols.append(qlist(f[f"x{k}"]))
        k += 1
    X = [([F(1)] if f.get("intercept") == "true" else []) + [c[i] for c in cols] for i in range(len(y))]
    return y, X


def statsmodels_ref(row):
    import numpy as np
    import statsmodels.api as sm
    y, X = model_design(row)
    res = sm.OLS(np.array([float(v) for v in y]), np.array([[float(v) for v in r] for r in X])).fit()
    out = {}
    for i in range(len(X[0])):
        out[f"coef.{i}"] = res.params[i]
        out[f"se.{i}"] = res.bse[i]
        out[f"t.{i}"] = res.tvalues[i]
        out[f"p.{i}"] = res.pvalues[i]
    out.update(rsquared=res.rsquared, rsquared_adj=res.rsquared_adj, fvalue=res.fvalue, f_pvalue=res.f_pvalue,
               llf=res.llf, aic=res.aic, bic=res.bic)
    return {k: float(v) for k, v in out.items()}


# ───────────────────────────── mpmath ─────────────────────────────

def ncdf(z):
    return mp.erfc(-z / mp.sqrt(2)) / 2


def _ibeta_cf(a, b, x, max_steps=10 ** 6):
    """I_x(a, b) for an exact rational x by the continued fraction (DLMF 8.17.22, modified Lentz): fast
    below (a+1)/(a+b+2), slower (but convergent) above; the prefactor from ln x and ln(1 − x) of the
    exact x.  Used where mpmath's hypergeometric route gives up, returns 0 for a tiny tail, or loses
    digits for an argument near 1."""
    y = 1 - x
    a, b, xm = M(a), M(b), M(x)
    eps = mp.mpf(2) ** (-mp.mp.prec - 8)
    tiny = mp.mpf(2) ** (-10 * mp.mp.prec)
    c, d = mp.mpf(1), 1 - (a + b) * xm / (a + 1)
    d = 1 / (d if d != 0 else tiny)
    h = d
    for m in range(1, max_steps):
        for num in (m * (b - m) * xm / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * xm / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + num * d
            d = 1 / (d if d != 0 else tiny)
            c = 1 + num / c
            c = c if c != 0 else tiny
            h *= d * c
        if abs(d * c - 1) < eps:
            break
    else:
        raise mp.libmp.NoConvergence("incomplete beta continued fraction")
    lpref = (a * mp.log(xm) + b * mp.log(M(y)) - mp.log(a)
             - (mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)))
    return mp.exp(lpref) * h


def _ibeta_direct(a, b, x):
    """I_x(a, b) computed directly, for x at most the switch point (a+1)/(a+b+2): mpmath's hypergeometric
    route where it answers (x ≤ ½), else the continued fraction (convergent there)."""
    if x <= F(1, 2):
        try:
            v = mp.betainc(M(a), M(b), 0, M(x), regularized=True)
            if v != 0:
                return v
        except mp.libmp.NoConvergence:
            pass
    return _ibeta_cf(a, b, x)


def ibeta_pair(a, b, x):
    """(I_x(a, b), 1 − I_x(a, b)) for exact rational x: the tail on the near side of the switch point
    (a+1)/(a+b+2) directly, the other as 1 minus it — unless that is small (below ¼), when it is also
    computed directly, by the continued fraction past its switch point (slower, but it converges):
    `1 − I_{1−x}(b, a)` cancels to 0 for `Beta(4.2·10⁶, 4.6·10⁻⁶⁵)` at `x = 1 − 5.4·10⁻⁸`, whose
    lower tail is `5.2·10⁻⁶⁵`."""
    if x <= 0: return mp.mpf(0), mp.mpf(1)
    if x >= 1: return mp.mpf(1), mp.mpf(0)
    near_lower = x <= (a + 1) / (a + b + 2)
    if near_lower:
        lo = _ibeta_direct(a, b, x)
        up = 1 - lo
    else:
        up = _ibeta_direct(b, a, 1 - x)
        lo = 1 - up
    # A bounded refinement: past its switch point the fraction may need very many steps (a tiny shape);
    # then the difference stands and the two-precision check decides whether it is trusted.
    try:
        if near_lower and up < mp.mpf(1) / 4:
            up = _ibeta_cf(b, a, 1 - x, max_steps=20_000)
        elif not near_lower and lo < mp.mpf(1) / 4:
            lo = _ibeta_cf(a, b, x, max_steps=20_000)
    except mp.libmp.NoConvergence:
        pass
    return lo, up


def ibeta(a, b, x):
    """I_x(a, b) for exact rational x in [0, 1]."""
    return ibeta_pair(a, b, x)[0]


def ibetac(a, b, x):
    """1 − I_x(a, b) = I_{1−x}(b, a)."""
    return ibeta_pair(a, b, x)[1]


def _gamma_pq(s, x):
    """(P(s, x), Q(s, x)) by the series (x < s + 1) or Legendre's continued fraction, each tail directly
    and the other as 1 minus it: the fallback where mpmath's own route gives up (s past ~10⁷)."""
    s, xm = M(s), M(x)
    eps = mp.mpf(2) ** (-mp.mp.prec - 8)
    lpref = s * mp.log(xm) - xm - mp.loggamma(s)
    if xm < s + 1:
        term = tot = 1 / s
        n = 0
        while abs(term) > eps * abs(tot):
            n += 1
            term *= xm / (s + n)
            tot += term
            if n > 10 ** 8:
                raise mp.libmp.NoConvergence("incomplete gamma series")
        p = mp.exp(lpref) * tot
        return p, 1 - p
    tiny = mp.mpf(2) ** (-10 * mp.mp.prec)
    b = xm + 1 - s
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10 ** 8):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1 / (d if d != 0 else tiny)
        c = b + an / c
        c = c if c != 0 else tiny
        h *= d * c
        if abs(d * c - 1) < eps:
            break
    q = mp.exp(lpref) * h
    return 1 - q, q


def gammp(s, x):
    if x <= 0: return mp.mpf(0)
    try:
        return mp.gammainc(M(s), 0, M(x), regularized=True)
    except mp.libmp.NoConvergence:
        return _gamma_pq(s, x)[0]


def gammq(s, x):
    if x <= 0: return mp.mpf(1)
    try:
        return mp.gammainc(M(s), M(x), mp.inf, regularized=True)
    except mp.libmp.NoConvergence:
        return _gamma_pq(s, x)[1]


def disc_pmf(fam, P, k):
    """Exact or mp pmf at integer k."""
    if fam == "bernoulli":
        return P[0] if k == 1 else (1 - P[0]) if k == 0 else F(0)
    if fam == "binomial":
        n, p = int(P[0]), P[1]
        if k < 0 or k > n: return F(0)
        if n <= 200 and p.denominator < 10 ** 6:
            return F(math.comb(n, k)) * p ** k * (1 - p) ** (n - k)
        return mp.exp(mp.loggamma(n + 1) - mp.loggamma(k + 1) - mp.loggamma(n - k + 1)
                      + k * mp.log(M(p)) + (n - k) * mp.log(M(1 - p)))
    if fam == "poisson":
        if k < 0: return F(0)
        lam = M(P[0])
        return mp.exp(k * mp.log(lam) - lam - mp.loggamma(k + 1))
    if fam == "geometric":
        if k < 1: return F(0)
        if k <= 64:
            return (1 - P[0]) ** (k - 1) * P[0]
        return mp.exp((k - 1) * mp.log(M(1 - P[0]))) * M(P[0])
    if fam == "negative_binomial":
        if k < 0: return F(0)
        r, p = M(P[0]), P[1]
        return mp.exp(mp.loggamma(k + r) - mp.loggamma(r) - mp.loggamma(k + 1)) * M(p) ** r * M(1 - p) ** k
    if fam == "hypergeometric":
        N, K, n = (int(v) for v in P)
        if k < max(0, n + K - N) or k > min(n, K): return F(0)
        return F(math.comb(K, k) * math.comb(N - K, n - k), math.comb(N, n))
    if fam == "discrete_uniform":
        a, b = int(P[0]), int(P[1])
        return F(1, b - a + 1) if a <= k <= b else F(0)
    if fam == "finite":
        return sum((p for v, p in P if v == k), F(0))
    raise KeyError(fam)


def disc_support(fam, P):
    """Finite (lo, hi) or (lo, None)."""
    if fam == "bernoulli": return 0, 1
    if fam == "binomial": return 0, int(P[0])
    if fam in ("poisson", "negative_binomial"): return 0, None
    if fam == "geometric": return 1, None
    if fam == "hypergeometric":
        N, K, n = (int(v) for v in P)
        return max(0, n + K - N), min(n, K)
    if fam == "discrete_uniform": return int(P[0]), int(P[1])
    raise KeyError(fam)


def disc_cdf_sf(fam, P, x):
    k = math.floor(x)
    if fam == "finite":
        c = sum((p for v, p in P if v <= x), F(0))
        return c, 1 - c
    lo, hi = disc_support(fam, P)
    if k < lo: return F(0), F(1)
    if hi is not None and k >= hi: return F(1), F(0)
    if fam == "poisson":
        return gammq(k + 1, P[0]), gammp(k + 1, P[0])
    if fam == "geometric":
        s = (1 - P[0]) ** k if k <= 64 else mp.exp(k * mp.log(M(1 - P[0])))
        return 1 - s, s
    if fam == "negative_binomial":
        r, p = P
        return ibeta_pair(r, k + 1, p)
    if fam == "binomial" and (int(P[0]) > 200 or P[1].denominator >= 10 ** 6):
        n, p = int(P[0]), P[1]
        return ibeta_pair(n - k, k + 1, 1 - p)
    c = sum((disc_pmf(fam, P, j) for j in range(lo, k + 1)), F(0))
    s = sum((disc_pmf(fam, P, j) for j in range(k + 1, hi + 1)), F(0))
    return c, s


def cont_pdf(fam, P, x):
    X = M(x)
    p = [M(v) for v in P]
    if fam == "normal": return mp.npdf(M((x - P[0]) / P[1])) / p[1]
    if fam == "uniform": return 1 / (p[1] - p[0]) if P[0] <= x <= P[1] else mp.mpf(0)
    if fam == "exponential": return p[0] * mp.exp(-p[0] * X) if x >= 0 else mp.mpf(0)
    if fam in ("gamma", "chi_squared"):
        k, th = (p[0], p[1]) if fam == "gamma" else (p[0] / 2, mp.mpf(2))
        if x < 0: return mp.mpf(0)
        if x == 0: return mp.inf if k < 1 else (1 / th if k == 1 else mp.mpf(0))
        return mp.exp((k - 1) * mp.log(X) - X / th - mp.loggamma(k) - k * mp.log(th))
    if fam == "beta":
        a, b = p
        if x < 0 or x > 1: return mp.mpf(0)
        if x == 0: return mp.inf if a < 1 else (b if a == 1 else mp.mpf(0))
        if x == 1: return mp.inf if b < 1 else (a if b == 1 else mp.mpf(0))
        return mp.exp((a - 1) * mp.log(X) + (b - 1) * mp.log(M(1 - x)) - mp.log(mp.beta(a, b)))
    # (x − loc)/scale exactly first: x and loc of 300 digits differing by 2 cancel in mpf.
    if fam == "cauchy": return 1 / (mp.pi * p[1] * (1 + M((x - P[0]) / P[1]) ** 2))
    if fam == "laplace": return mp.exp(-abs(M((x - P[0]) / P[1]))) / (2 * p[1])
    if fam == "logistic":
        e = mp.exp(-abs(M((x - P[0]) / P[1])))
        return e / (p[1] * (1 + e) ** 2)
    if fam == "log_normal":
        if x <= 0: return mp.mpf(0)
        return mp.exp(-(mp.log(X) - p[0]) ** 2 / (2 * p[1] ** 2)) / (X * p[1] * mp.sqrt(2 * mp.pi))
    if fam == "student_t":
        n = p[0]
        return mp.exp(mp.loggamma((n + 1) / 2) - mp.loggamma(n / 2)) / mp.sqrt(n * mp.pi) * (1 + X * X / n) ** (-(n + 1) / 2)
    if fam == "f":
        d1, d2 = p
        if x < 0: return mp.mpf(0)
        if x == 0: return mp.inf if d1 < 2 else (mp.mpf(1) if d1 == 2 else mp.mpf(0))
        return mp.exp((d1 / 2) * mp.log(d1 * X) + (d2 / 2) * mp.log(d2) - ((d1 + d2) / 2) * mp.log(d1 * X + d2)
                      - mp.log(X) - mp.log(mp.beta(d1 / 2, d2 / 2)))
    if fam == "weibull":
        lam, k = p
        if x < 0: return mp.mpf(0)
        if x == 0: return mp.inf if k < 1 else (1 / lam if k == 1 else mp.mpf(0))
        return (k / lam) * (X / lam) ** (k - 1) * mp.exp(-(X / lam) ** k)
    if fam == "pareto":
        xm, a = p
        return a * xm ** a / X ** (a + 1) if x >= P[0] else mp.mpf(0)
    if fam == "triangular":
        a, b, c = P
        if x < a or x > b: return mp.mpf(0)
        if x < c or (x == c and c > a):
            return M(2 * (x - a) / ((b - a) * (c - a)))
        return M(2 * (b - x) / ((b - a) * (b - c))) if b > c else M(F(2) / (b - a))
    raise KeyError(fam)


def cont_cdf_sf(fam, P, x):
    X = M(x)
    p = [M(v) for v in P]
    one, zero = mp.mpf(1), mp.mpf(0)
    if fam == "normal":
        z = M((x - P[0]) / P[1])
        return ncdf(z), ncdf(-z)
    if fam == "uniform":
        a, b = P
        if x <= a: return zero, one
        if x >= b: return one, zero
        return M((x - a) / (b - a)), M((b - x) / (b - a))
    if fam == "exponential":
        if x <= 0: return zero, one
        return -mp.expm1(-p[0] * X), mp.exp(-p[0] * X)
    if fam == "gamma":
        return gammp(P[0], x / P[1]), gammq(P[0], x / P[1])
    if fam == "chi_squared":
        return gammp(P[0] / 2, x / 2), gammq(P[0] / 2, x / 2)
    if fam == "beta":
        return ibeta_pair(P[0], P[1], x)
    if fam == "cauchy":
        z = M((x - P[0]) / P[1])
        return mp.atan2(1, -z) / mp.pi, mp.atan2(1, z) / mp.pi
    if fam == "laplace":
        z = M((x - P[0]) / P[1])
        if z < 0: return mp.exp(z) / 2, 1 - mp.exp(z) / 2
        return 1 - mp.exp(-z) / 2, mp.exp(-z) / 2
    if fam == "logistic":
        z = M((x - P[0]) / P[1])
        return 1 / (1 + mp.exp(-z)), 1 / (1 + mp.exp(z))
    if fam == "log_normal":
        if x <= 0: return zero, one
        z = (mp.log(X) - p[0]) / p[1]
        return ncdf(z), ncdf(-z)
    if fam == "student_t":
        n = P[0]
        if x == 0: return one / 2, one / 2
        w = n / (n + x * x)
        tail, rest = ibeta_pair(n / 2, F(1, 2), w)
        tail /= 2
        return (tail, 1 - tail) if x < 0 else (1 - tail, tail)
    if fam == "f":
        if x <= 0: return zero, one
        d1, d2 = P
        z = d1 * x / (d1 * x + d2)
        return ibeta_pair(d1 / 2, d2 / 2, z)
    if fam == "weibull":
        if x <= 0: return zero, one
        y = (X / p[0]) ** p[1]
        return -mp.expm1(-y), mp.exp(-y)
    if fam == "pareto":
        if x <= P[0]: return zero, one
        s = (p[0] / X) ** p[1]
        return -mp.expm1(p[1] * mp.log(p[0] / X)), s
    if fam == "triangular":
        a, b, c = P
        if x <= a: return zero, one
        if x >= b: return one, zero
        if x <= c and c > a:
            v = (x - a) ** 2 / ((b - a) * (c - a))
            return M(v), M(1 - v)
        v = (b - x) ** 2 / ((b - a) * (b - c))
        return M(1 - v), M(v)
    raise KeyError(fam)


def closed_quantile(fam, P, p):
    """The quantile at an exact level p, for the families with a closed form."""
    pm = [M(v) for v in P]
    lev = M(p)
    if fam == "normal": return pm[0] + pm[1] * mp.sqrt(2) * mp.erfinv(M(2 * p - 1))
    if fam == "log_normal": return mp.exp(pm[0] + pm[1] * mp.sqrt(2) * mp.erfinv(M(2 * p - 1)))
    if fam == "uniform": return M(P[0] + p * (P[1] - P[0]))
    if fam == "exponential": return -mp.log1p(-lev) / pm[0]
    if fam == "cauchy": return pm[0] + pm[1] * mp.tan(mp.pi * M(p - F(1, 2)))
    if fam == "laplace":
        return pm[0] + pm[1] * mp.log(M(2 * p)) if p < F(1, 2) else pm[0] - pm[1] * mp.log(M(2 - 2 * p))
    if fam == "logistic": return pm[0] + pm[1] * mp.log(M(p / (1 - p)))
    if fam == "weibull": return pm[0] * (-mp.log1p(-lev)) ** (1 / pm[1])
    if fam == "pareto": return pm[0] * M(1 - p) ** (-1 / pm[1])
    if fam == "triangular":
        a, b, c = P
        if p < (c - a) / (b - a):
            return M(a) + mp.sqrt(M(p * (b - a) * (c - a)))
        return M(b) - mp.sqrt(M((1 - p) * (b - a) * (b - c)))
    return None


def _moments_from_raw(m1, m2, m3, m4):
    var = m2 - m1 ** 2
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return m1, var, c3 / var ** mp.mpf(1.5), c4 / var ** 2


def moments(fam, P):
    """(mean, variance, skewness, kurtosis); "inf"/"-inf"/"nan" by symplex's convention: a moment
    divergent in one heavy tail is ±∞, divergent in both (or with an undefined mean) NaN; skewness and
    kurtosis are ±∞ where the variance is finite and the third / fourth moment diverges in one tail."""
    p = [M(v) for v in P] if fam != "finite" else None
    if fam == "normal": return p[0], p[1] ** 2, mp.mpf(0), mp.mpf(3)
    if fam == "uniform": return (p[0] + p[1]) / 2, (p[1] - p[0]) ** 2 / 12, mp.mpf(0), mp.mpf(9) / 5
    if fam == "exponential": return 1 / p[0], 1 / p[0] ** 2, mp.mpf(2), mp.mpf(9)
    if fam == "gamma": return p[0] * p[1], p[0] * p[1] ** 2, 2 / mp.sqrt(p[0]), 3 + 6 / p[0]
    if fam == "chi_squared": return p[0], 2 * p[0], mp.sqrt(8 / p[0]), 3 + 12 / p[0]
    if fam == "beta":
        a, b = p
        s = a + b
        return (a / s, a * b / (s ** 2 * (s + 1)), 2 * (b - a) * mp.sqrt(s + 1) / ((s + 2) * mp.sqrt(a * b)),
                3 + 6 * ((a - b) ** 2 * (s + 1) - a * b * (s + 2)) / (a * b * (s + 2) * (s + 3)))
    if fam == "cauchy": return "nan", "nan", "nan", "nan"
    if fam == "laplace": return p[0], 2 * p[1] ** 2, mp.mpf(0), mp.mpf(6)
    if fam == "logistic": return p[0], p[1] ** 2 * mp.pi ** 2 / 3, mp.mpf(0), mp.mpf(21) / 5
    if fam == "log_normal":
        raw = [mp.exp(n * p[0] + n * n * p[1] ** 2 / 2) for n in (1, 2, 3, 4)]
        return _moments_from_raw(*raw)
    if fam == "student_t":
        n = P[0]
        mean = mp.mpf(0) if n > 1 else "nan"
        var = M(n / (n - 2)) if n > 2 else ("inf" if n > 1 else "nan")
        skew = mp.mpf(0) if n > 3 else "nan"
        kurt = 3 + M(F(6) / (n - 4)) if n > 4 else ("inf" if n > 2 else "nan")
        return mean, var, skew, kurt
    if fam == "f":
        d1, d2 = P
        def raw(n):
            return mp.exp(n * mp.log(M(d2 / d1)) + mp.loggamma(M(d1 / 2) + n) + mp.loggamma(M(d2 / 2) - n)
                          - mp.loggamma(M(d1 / 2)) - mp.loggamma(M(d2 / 2)))
        # One heavy (right) tail: a divergent moment is +∞, also with an infinite mean.
        mean = raw(1) if d2 > 2 else "inf"
        if d2 <= 4:
            return mean, "inf", "nan", "nan"
        m1, var, sk, ku = _moments_from_raw(raw(1), raw(2), raw(3) if d2 > 6 else 0, raw(4) if d2 > 8 else 0)
        return m1, var, sk if d2 > 6 else "inf", ku if d2 > 8 else "inf"
    if fam == "weibull":
        lam, k = p
        raw = [lam ** n * mp.gamma(1 + n / k) for n in (1, 2, 3, 4)]
        return _moments_from_raw(*raw)
    if fam == "pareto":
        xm, a = P
        def raw(n):
            return M(a * xm ** n / (a - n))
        mean = raw(1) if a > 1 else "inf"
        if a <= 2:
            return mean, "inf", "nan", "nan"
        m1, var, sk, ku = _moments_from_raw(raw(1), raw(2), raw(3) if a > 3 else 0, raw(4) if a > 4 else 0)
        return m1, var, sk if a > 3 else "inf", ku if a > 4 else "inf"
    if fam == "triangular":
        a, b, c = P
        mean = (a + b + c) / 3
        var = (a * a + b * b + c * c - a * b - a * c - b * c) / 18
        num = mp.sqrt(2) * M((a + b - 2 * c) * (2 * a - b - c) * (a - 2 * b + c))
        skew = num / (5 * M(a * a + b * b + c * c - a * b - a * c - b * c) ** mp.mpf(1.5))
        return M(mean), M(var), skew, mp.mpf(12) / 5
    if fam in DISC:
        return disc_moments(fam, P)
    raise KeyError(fam)


def disc_moments(fam, P):
    """(mean, variance, skewness, kurtosis): rational in the parameters where they are (exact Fractions,
    so that `1 − p` for `p = 1 − 10⁻¹²` loses nothing), with the one square root in mpmath."""
    def out(mean, var, skew_num, kurt):
        if var == 0:
            return M(mean), mp.mpf(0), "nan", "nan"
        return M(mean), M(var), M(skew_num) / mp.sqrt(M(var)), M(kurt)
    if fam == "bernoulli":
        q = P[0]
        v = q * (1 - q)
        return out(q, v, 1 - 2 * q, 3 + (1 - 6 * v) / v if v else 0)
    if fam == "binomial":
        n, q = P
        v = n * q * (1 - q)
        return out(n * q, v, 1 - 2 * q, 3 + (1 - 6 * q * (1 - q)) / v if v else 0)
    if fam == "poisson":
        lam = P[0]
        return out(lam, lam, F(1), 3 + 1 / lam)
    if fam == "geometric":
        q = P[0]
        if q == 1:
            return out(F(1), F(0), 0, 0)
        return M(1 / q), M((1 - q) / q ** 2), M(2 - q) / mp.sqrt(M(1 - q)), M(9 + q ** 2 / (1 - q))
    if fam == "negative_binomial":
        r, q = P
        return (M(r * (1 - q) / q), M(r * (1 - q) / q ** 2), M(2 - q) / mp.sqrt(M(r * (1 - q))),
                M(3 + 6 / r + q ** 2 / (r * (1 - q))))
    if fam == "finite":
        atoms = P
    else:
        lo, hi = disc_support(fam, P)
        atoms = [(F(k), disc_pmf(fam, P, k)) for k in range(lo, hi + 1)]
    m1 = sum((v * w for v, w in atoms), F(0))
    c = [sum(((v - m1) ** j * w for v, w in atoms), F(0)) for j in (2, 3, 4)]
    if c[0] == 0:
        # A point mass: skewness and kurtosis are 0/0, undefined.
        return M(m1), mp.mpf(0), "nan", "nan"
    return M(m1), M(c[0]), M(c[1]) / M(c[0]) ** mp.mpf(1.5), M(c[2] / c[0] ** 2)


def entropy(fam, P):
    p = [M(v) for v in P] if fam != "finite" else None
    psi = mp.digamma
    if fam == "normal": return mp.log(2 * mp.pi * mp.e * p[1] ** 2) / 2
    if fam == "uniform": return mp.log(p[1] - p[0])
    if fam == "exponential": return 1 - mp.log(p[0])
    if fam == "gamma":
        k, th = p
        return k + mp.log(th) + mp.loggamma(k) + (1 - k) * psi(k)
    if fam == "chi_squared":
        h = p[0] / 2
        return h + mp.log(2) + mp.loggamma(h) + (1 - h) * psi(h)
    if fam == "beta":
        a, b = p
        return mp.log(mp.beta(a, b)) - (a - 1) * psi(a) - (b - 1) * psi(b) + (a + b - 2) * psi(a + b)
    if fam == "cauchy": return mp.log(4 * mp.pi * p[1])
    if fam == "laplace": return 1 + mp.log(2 * p[1])
    if fam == "logistic": return mp.log(p[1]) + 2
    if fam == "log_normal": return p[0] + mp.log(2 * mp.pi * mp.e * p[1] ** 2) / 2
    if fam == "student_t":
        n = p[0]
        return (n + 1) / 2 * (psi((n + 1) / 2) - psi(n / 2)) + mp.log(mp.sqrt(n) * mp.beta(n / 2, mp.mpf(1) / 2))
    if fam == "f":
        d1, d2 = p
        return (mp.log(d2 / d1) + mp.log(mp.beta(d1 / 2, d2 / 2)) + (1 - d1 / 2) * psi(d1 / 2)
                - (1 + d2 / 2) * psi(d2 / 2) + (d1 + d2) / 2 * psi((d1 + d2) / 2))
    if fam == "weibull":
        lam, k = p
        return mp.euler * (1 - 1 / k) + mp.log(lam / k) + 1
    if fam == "pareto":
        xm, a = p
        return mp.log(xm / a) + 1 / a + 1
    if fam == "triangular":
        return mp.mpf(1) / 2 + mp.log(M(P[1] - P[0]) / 2)
    if fam == "geometric":
        q = p[0]
        if q == 1: return mp.mpf(0)
        return (-(1 - q) * mp.log(1 - q) - q * mp.log(q)) / q
    if fam in ("bernoulli", "binomial", "hypergeometric", "discrete_uniform", "finite", "poisson",
               "negative_binomial"):
        if fam == "finite":
            ws = [w for _, w in P]
        elif fam in ("poisson", "negative_binomial"):
            mu, var, _, _ = disc_moments(fam, P)
            lo = max(0, int(mu - 40 * mp.sqrt(var) - 50))
            hi = int(mu + 40 * mp.sqrt(var) + 50)
            if fam == "negative_binomial" and P[1] < 1:
                # The tail decays like (1 − p)^k whatever the mean: with r = 5e-30, p = 0.25 the
                # window above stopped at k = 50, where (3/4)^k is 6e-7 (an entropy 3e-8 off).
                hi += int(mp.mpf(80) * mp.log(10) / -mp.log(M(1 - P[1]))) + 1
            ws = [disc_pmf(fam, P, k) for k in range(lo, hi + 1)]
        else:
            lo, hi = disc_support(fam, P)
            if fam == "binomial" and hi > 5000:
                mu, var, _, _ = disc_moments(fam, P)
                lo, hi = max(lo, int(mu - 40 * mp.sqrt(var) - 50)), min(hi, int(mu + 40 * mp.sqrt(var) + 50))
            ws = [disc_pmf(fam, P, k) for k in range(lo, hi + 1)]
        return -mp.fsum(M(w) * mp.log(M(w)) for w in ws if w != 0)
    raise KeyError(fam)


def mgf(fam, P, t):
    """E[e^{tX}] at an exact or complex t (an mpf/mpc), "inf" beyond the abscissa."""
    p = [M(v) for v in P] if fam != "finite" else None
    real = not isinstance(t, mp.mpc)
    if fam == "normal": return mp.exp(p[0] * t + p[1] ** 2 * t * t / 2)
    if fam == "uniform":
        if t == 0: return mp.mpf(1)
        return (mp.exp(t * p[1]) - mp.exp(t * p[0])) / (t * (p[1] - p[0]))
    if fam == "exponential":
        if real and t >= p[0]: return "inf"
        return p[0] / (p[0] - t)
    if fam == "gamma":
        if real and t * p[1] >= 1: return "inf"
        return (1 - p[1] * t) ** (-p[0])
    if fam == "chi_squared":
        if real and 2 * t >= 1: return "inf"
        return (1 - 2 * t) ** (-p[0] / 2)
    if fam == "laplace":
        if real and abs(t) * p[1] >= 1: return "inf"
        return mp.exp(p[0] * t) / (1 - p[1] ** 2 * t * t)
    if fam == "logistic":
        if real and abs(t) * p[1] >= 1: return "inf"
        st = p[1] * t
        return mp.exp(p[0] * t) * mp.pi * st / mp.sin(mp.pi * st) if st != 0 else mp.mpf(1)
    if fam == "cauchy":
        if real: return mp.mpf(1) if t == 0 else "inf"
        s = t.imag
        return mp.exp(1j * s * p[0] - p[1] * abs(s))
    if fam == "triangular":
        # Closed forms (mp.quad of e^{tx} over a short piece agreed with itself at two precisions
        # and was off by 4e-12).
        a, b, c = p
        if t == 0:
            return mp.mpf(1)
        ea, eb, ec = mp.exp(t * a), mp.exp(t * b), mp.exp(t * c)
        if a < c < b:
            return 2 * ((b - c) * ea - (b - a) * ec + (c - a) * eb) / ((b - a) * (c - a) * (b - c) * t * t)
        w = b - a
        if c == a:
            return 2 / w ** 2 * ((eb - ea) / t ** 2 - w * ea / t)
        return 2 / w ** 2 * (w * eb / t - (eb - ea) / t ** 2)
    if fam == "bernoulli": return 1 - p[0] + p[0] * mp.exp(t)
    if fam == "binomial": return (1 - p[1] + p[1] * mp.exp(t)) ** p[0]
    if fam == "poisson": return mp.exp(p[0] * (mp.exp(t) - 1))
    if fam == "geometric":
        q = p[0]
        if real and (1 - q) * mp.exp(t) >= 1: return "inf"
        return q * mp.exp(t) / (1 - (1 - q) * mp.exp(t))
    if fam == "negative_binomial":
        r, q = p
        if real and (1 - q) * mp.exp(t) >= 1: return "inf"
        return (q / (1 - (1 - q) * mp.exp(t))) ** r
    if fam == "discrete_uniform":
        a, b = int(P[0]), int(P[1])
        return mp.fsum(mp.exp(t * k) for k in range(a, b + 1)) / (b - a + 1)
    if fam == "finite":
        return mp.fsum(M(w) * mp.exp(t * M(v)) for v, w in P)
    return None


def p_tail_t(tstat, df, alt):
    t, n = mp.mpf(tstat), mp.mpf(df)
    def sf(u):
        w = n / (n + u * u)
        tail = mp.betainc(n / 2, mp.mpf(1) / 2, 0, w, regularized=True) / 2
        return tail if u > 0 else 1 - tail
    if alt == "greater": return sf(t)
    if alt == "less": return sf(-t)
    return min(mp.mpf(1), 2 * sf(abs(t)))


def p_tail_z(z, alt):
    z = mp.mpf(z)
    if alt == "greater": return ncdf(-z)
    if alt == "less": return ncdf(z)
    return mp.erfc(abs(z) / mp.sqrt(2))
