import numpy as np
import matplotlib.pyplot as plt

def build_clinical_matrix(cov_row_np, cov_names):
    """
    Returns a 1D array of clinical covariates for multivariable Cox.
    Suggested set for PSMA:
      age, log_psa_at_scan (fallback log_psa_pre/initial),
      grade_group, t_ord, BMI, indication (one-hot: primary/recurrence/metastatic),
      pre_androgen_targeted, pre_cytotoxic

        Note:
            We intentionally EXCLUDE any post-PSMA covariates (e.g., post_local, post_focal,
            post_at, post_cyto) to avoid including variables that are downstream of the PSMA
            imaging itself.
    """
    def get(nm, default=np.nan):
        try:
            j = cov_names.index(nm); return cov_row_np[j]
        except ValueError:
            return default

    # PSA at scan
    lpsa = get('log_psa_at_scan')
    if not np.isfinite(lpsa):
        lpsa = get('log_psa_pre')
    if not np.isfinite(lpsa):
        lpsa = get('log_psa_initial')

    age = get('age'); gg = get('grade_group'); tord = get('t_ord'); bmi = get('bmi')

    ind_primary    = get('ind_primary', 0.0)
    ind_recurrence = get('ind_recurrence', 0.0)
    ind_metastatic = get('ind_metastatic', 0.0)

    pre_at   = get('pre_at', 0.0)
    pre_cyto = get('pre_cyto', 0.0)

    row = np.array([age, lpsa, gg, tord, bmi,
                    ind_primary, ind_recurrence, ind_metastatic,
                    pre_at, pre_cyto], dtype=float)

    # replace NaNs with column means (computed later in zscore)
    return row

def _cox_sort_by_time(time, event, X=None):
    idx = np.argsort(time)  # ascending time
    time_s = time[idx]
    event_s = event[idx]
    if X is None:
        return time_s, event_s, None, idx
    return time_s, event_s, X[idx, ...], idx

def _cox_partial_grad_hess(beta, X, time, event, l2=0.0):
    # Returns gradient and Fisher information (positive semidefinite)
    if X.ndim == 1:
        X = X[:, None]
    N, P = X.shape
    xb = X @ beta
    r  = np.exp(xb)
    order = np.argsort(time)   # ascending
    time = time[order]
    event = event[order]
    X = X[order, :]
    r = r[order]

    cr = np.cumsum(r[::-1])[::-1]                                  # S0
    cX = np.cumsum((r[:, None] * X)[::-1, :], axis=0)[::-1, :]     # S1

    g = np.zeros(P)
    I = np.zeros((P, P))  # Fisher information

    i = 0
    while i < N:
        t = time[i]
        j = i
        while j < N and time[j] == t:
            j += 1
        d = int(event[i:j].sum())
        if d > 0:
            s0 = cr[i]
            s1 = cX[i, :]
            mu = s1 / max(s0, 1e-12)
            xe_sum = X[i:j, :][event[i:j] == 1].sum(axis=0)
            g += xe_sum - d * mu

            X_risk = X[i:, :]
            r_risk = r[i:]
            s2 = (X_risk.T * r_risk) @ X_risk / max(s0, 1e-12)
            I += d * (s2 - np.outer(mu, mu))
        i = j

    # Ridge penalty for penalized log-likelihood: l(beta) - (l2/2)||beta||^2
    if l2 > 0:
        g -= l2 * beta
        I += l2 * np.eye(P)
    return g, I

def cox_fit(X, time, event, l2=1e-4, max_iter=50, tol=1e-6,
            return_diag=False, cond_thresh=1e8, l2_max=1e-2):
    """
    Newton/Fisher scoring for Cox PH with adaptive ridge if I is ill-conditioned.
    Returns: beta[P], se[P], z[P], p[P] (and diag if return_diag)
    """
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(X), axis=1)
    X, time, event = X[mask], time[mask], event[mask]
    if X.shape[0] < 10 or event.sum() < 5:
        P = X.shape[1]
        out = (np.zeros(P), np.full(P, np.inf), np.zeros(P), np.ones(P))
        if return_diag:
            return (*out, dict(n_iter=0, converged=False, cond=np.nan, l2_used=l2, step_max=np.nan))
        return out

    time_s, event_s, X_s, _ = _cox_sort_by_time(time, event, X)
    P = X_s.shape[1]
    beta = np.zeros(P)
    last_step_max = np.nan
    last_cond = np.nan
    l2_used = float(l2)

    for it in range(max_iter):
        g, I = _cox_partial_grad_hess(beta, X_s, time_s, event_s, l2=l2_used)
        # Adaptive ridge to control conditioning of I
        try:
            last_cond = np.linalg.cond(I)
        except np.linalg.LinAlgError:
            last_cond = np.inf
        if not np.isfinite(last_cond) or last_cond > cond_thresh:
            # bump ridge up to l2_max
            bump = l2_used
            while bump < l2_max:
                bump = min(l2_max, max(bump * 10.0, 1e-6))
                I = I + (bump - l2_used) * np.eye(P)
                try:
                    last_cond = np.linalg.cond(I)
                except np.linalg.LinAlgError:
                    last_cond = np.inf
                if np.isfinite(last_cond) and last_cond <= cond_thresh:
                    l2_used = bump
                    break
            # if still bad, proceed with pinv solve below

        try:
            step = np.linalg.solve(I, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(I) @ g
        beta_new = beta + step
        last_step_max = float(np.max(np.abs(step)))
        beta = beta_new
        if last_step_max < tol:
            break

    # Variance from Fisher information at the solution (use final l2_used)
    _, Ifinal = _cox_partial_grad_hess(beta, X_s, time_s, event_s, l2=l2_used)
    try:
        var = np.linalg.inv(Ifinal)
    except np.linalg.LinAlgError:
        var = np.linalg.pinv(Ifinal)
    diagv = np.diag(var)
    diagv = np.where(diagv > 1e-12, diagv, 1e-12)
    se = np.sqrt(diagv)

    z = beta / se
    def norm_cdf_vec(x):
        x = np.asarray(x, dtype=float)
        try:
            from scipy.special import ndtr as _ndtr
            return _ndtr(x)
        except Exception:
            from math import erf, sqrt
            return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))
    p = 2.0 * (1.0 - norm_cdf_vec(np.abs(z)))  # keep two-sided

    if return_diag:
        return beta, se, z, p, dict(n_iter=it+1, converged=last_step_max < tol,
                                    cond=last_cond, l2_used=l2_used, step_max=last_step_max)
    return beta, se, z, p

def benjamini_hochberg(pvals, q=0.05):
    """BH-FDR that ignores NaNs and returns NaN q-values where p is NaN."""
    p = np.asarray(pvals, dtype=float)
    valid = np.isfinite(p)
    m = int(valid.sum())
    sig = np.zeros_like(p, dtype=bool)
    qvals = np.full_like(p, np.nan, dtype=float)
    if m == 0:
        return sig, qvals

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    thresh = q * (np.arange(1, m+1) / m)
    passed = ranked <= thresh
    if passed.any():
        k = int(np.max(np.where(passed)[0]))
        cutoff = ranked[k]
        sig_valid = p_valid <= cutoff
    else:
        sig_valid = np.zeros_like(p_valid, dtype=bool)

    # standard monotone q-values on valid entries only
    q_tmp = ranked * m / np.arange(1, m+1)
    q_tmp = np.minimum.accumulate(q_tmp[::-1])[::-1]
    out_valid = np.clip(q_tmp, 0, 1)
    qvals_valid = np.empty_like(p_valid)
    qvals_valid[order] = out_valid

    qvals[valid] = qvals_valid
    sig[valid] = sig_valid
    return sig, qvals

def concordance_index(time, event, risk):
    # Harrell’s C (naive O(N^2), fine for N~100-300)
    n = len(time)
    num = 0; den = 0
    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i] == 1:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return num / den if den > 0 else np.nan

# Helper: orthogonalize a vector against covariates (remove linear redundancy)
def orthogonalize_against(y_vec, Xmat):
    y = y_vec.reshape(-1, 1)
    try:
        coef, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
        resid = y - Xmat @ coef
        return resid.reshape(-1, 1)
    except Exception:
        return y

# ---------- Kaplan–Meier utilities ----------
def _km_step(time, event):
    """
    Compute Kaplan–Meier step times and survival estimates for one group.
    Inputs are 1D numpy arrays (time in days, event in {0,1}).
    Returns (times_sorted, survival_probs) with stepwise stairs plot style.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    # sort by time ascending
    order = np.argsort(t)
    t = t[order]
    e = e[order]
    # unique event times
    uniq = np.unique(t[e == 1])
    if uniq.size == 0:
        # no events: survival stays at 1
        return np.array([0.0], dtype=float), np.array([1.0], dtype=float)
    S = 1.0
    times_plot = [0.0]
    surv_plot = [1.0]
    # risk set counts are evaluated at each event time
    for ti in uniq:
        at_risk = np.sum(t >= ti)
        d_i = np.sum((t == ti) & (e == 1))
        if at_risk <= 0:
            continue
        frac = 1.0 - (d_i / float(at_risk))
        frac = max(min(frac, 1.0), 0.0)
        # step just before drop
        times_plot.append(ti)
        surv_plot.append(surv_plot[-1])
        # drop at event time
        S = S * frac
        times_plot.append(ti)
        surv_plot.append(S)
    return np.asarray(times_plot, dtype=float), np.asarray(surv_plot, dtype=float)

def plot_km_two_groups(time, event, group_bool, out_path, title='Kaplan–Meier: High vs Low risk', to_years=True, ci=True, alpha_band=0.20):
    """Plot KM curves for two groups defined by group_bool (True=High, False=Low) with optional Greenwood 95% CI bands."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    g = np.asarray(group_bool, dtype=bool)
    if time.shape[0] == 0 or (~np.isfinite(time)).all():
        return
    m_hi = g & np.isfinite(time) & np.isfinite(event)
    m_lo = (~g) & np.isfinite(time) & np.isfinite(event)
    if m_hi.sum() == 0 or m_lo.sum() == 0:
        return

    def km_with_var(t_in, e_in):
        t = np.asarray(t_in, dtype=float); e = np.asarray(e_in, dtype=int)
        ord_idx = np.argsort(t); t = t[ord_idx]; e = e[ord_idx]
        uniq = np.unique(t[e == 1])
        if uniq.size == 0:
            return np.array([0.0]), np.array([1.0]), np.array([0.0])
        S = 1.0
        times_plot = [0.0]; surv_plot = [1.0]
        var_terms_cum = []
        acc = 0.0
        for ti in uniq:
            at_risk = np.sum(t >= ti)
            d_i = np.sum((t == ti) & (e == 1))
            if at_risk <= 0:
                continue
            frac = 1.0 - d_i / float(at_risk)
            frac = max(min(frac, 1.0), 0.0)
            times_plot.append(ti); surv_plot.append(surv_plot[-1])
            S = S * frac
            times_plot.append(ti); surv_plot.append(S)
            # Greenwood accumulation
            if at_risk > d_i:
                acc += d_i / (at_risk * (at_risk - d_i))
            var_terms_cum.append(acc)
        times = np.asarray(times_plot, dtype=float)
        surv = np.asarray(surv_plot, dtype=float)
        # Map cumulative variance terms to survival steps (duplicate last for pre-drop segment)
        var_cum = np.zeros_like(surv)
        if len(var_terms_cum):
            # Each event produced two appended survival points; variance applies after the drop
            vi = 0
            for k in range(2, len(surv)):
                # After each drop (even indices after first two), advance variance index when surv decreases
                if surv[k] != surv[k-1] and vi < len(var_terms_cum):
                    vi_curr = var_terms_cum[vi]; vi += 1
                var_cum[k] = vi_curr if 'vi_curr' in locals() else 0.0
            var_cum[0:2] = 0.0
        var_surv = (surv ** 2) * var_cum  # Greenwood formula
        return times, surv, var_surv

    t_hi, s_hi, v_hi = km_with_var(time[m_hi], event[m_hi])
    t_lo, s_lo, v_lo = km_with_var(time[m_lo], event[m_lo])

    if to_years:
        t_hi = t_hi / 365.25; t_lo = t_lo / 365.25; xlab = 'Time (years)'
    else:
        xlab = 'Time (days)'

    plt.figure(figsize=(6.5, 5.0), dpi=140)
    plt.step(t_hi, s_hi, where='post', color='#d62728', linewidth=2.0, label=f'High (N={m_hi.sum()}, events={int(event[m_hi].sum())})')
    plt.step(t_lo, s_lo, where='post', color='#1f77b4', linewidth=2.0, label=f'Low (N={m_lo.sum()}, events={int(event[m_lo].sum())})')

    if ci:
        se_hi = np.sqrt(np.clip(v_hi, 0.0, None))
        se_lo = np.sqrt(np.clip(v_lo, 0.0, None))
        ci_hi_low = np.clip(s_hi - 1.96 * se_hi, 0.0, 1.0)
        ci_hi_high = np.clip(s_hi + 1.96 * se_hi, 0.0, 1.0)
        ci_lo_low = np.clip(s_lo - 1.96 * se_lo, 0.0, 1.0)
        ci_lo_high = np.clip(s_lo + 1.96 * se_lo, 0.0, 1.0)
        plt.fill_between(t_hi, ci_hi_low, ci_hi_high, color='#d62728', alpha=alpha_band, linewidth=0)
        plt.fill_between(t_lo, ci_lo_low, ci_lo_high, color='#1f77b4', alpha=alpha_band, linewidth=0)

    plt.ylim(0.0, 1.0); plt.xlim(left=0.0)
    plt.grid(alpha=0.25, linestyle='--')
    plt.xlabel(xlab); plt.ylabel('Survival probability'); plt.title(title)
    plt.legend(loc='best', frameon=False); plt.tight_layout()
    try:
        plt.savefig(out_path)
    finally:
        plt.close()

def plot_km_three_groups(time, event, risk, out_path,
                         title='Kaplan–Meier: Low vs Medium vs High risk (tertiles)',
                         to_years=True, ci=True, alpha_band=0.20):
    """Tertile-based three-group KM with optional Greenwood 95% CI bands.
    risk: 1D array of prognostic index used to split into low/medium/high.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    m_valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(r)
    if m_valid.sum() == 0:
        return
    r_valid = r[m_valid]
    # Tertile thresholds; guard for ties
    q1, q2 = np.nanquantile(r_valid, [1/3, 2/3])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        # fallback to median split if tertiles collapse
        q1 = q2 = np.nanmedian(r_valid)
    m_low = (r <= q1) & m_valid
    m_high = (r >= q2) & m_valid
    m_med = (~m_low & ~m_high) & m_valid
    if m_low.sum() == 0 or m_med.sum() == 0 or m_high.sum() == 0:
        return

    def km_with_var(t_in, e_in):
        t = np.asarray(t_in, dtype=float); e = np.asarray(e_in, dtype=int)
        ord_idx = np.argsort(t); t = t[ord_idx]; e = e[ord_idx]
        uniq = np.unique(t[e == 1])
        if uniq.size == 0:
            return np.array([0.0]), np.array([1.0]), np.array([0.0])
        S = 1.0
        times_plot = [0.0]; surv_plot = [1.0]
        var_terms_cum = []
        acc = 0.0
        for ti in uniq:
            at_risk = np.sum(t >= ti); d_i = np.sum((t == ti) & (e == 1))
            if at_risk <= 0:
                continue
            times_plot.append(ti); surv_plot.append(surv_plot[-1])
            S = S * (1.0 - d_i / float(at_risk))
            times_plot.append(ti); surv_plot.append(S)
            if at_risk > d_i:
                acc += d_i / (at_risk * (at_risk - d_i))
            var_terms_cum.append(acc)
        times = np.asarray(times_plot, dtype=float)
        surv = np.asarray(surv_plot, dtype=float)
        var_cum = np.zeros_like(surv)
        if len(var_terms_cum):
            vi = 0
            for k in range(2, len(surv)):
                if surv[k] != surv[k-1] and vi < len(var_terms_cum):
                    vi_curr = var_terms_cum[vi]; vi += 1
                var_cum[k] = vi_curr if 'vi_curr' in locals() else 0.0
            var_cum[0:2] = 0.0
        var_surv = (surv ** 2) * var_cum
        return times, surv, var_surv

    t_lo, s_lo, v_lo = km_with_var(time[m_low], event[m_low])
    t_md, s_md, v_md = km_with_var(time[m_med], event[m_med])
    t_hi, s_hi, v_hi = km_with_var(time[m_high], event[m_high])

    if to_years:
        t_lo, t_md, t_hi = t_lo/365.25, t_md/365.25, t_hi/365.25
        xlab = 'Time (years)'
    else:
        xlab = 'Time (days)'

    plt.figure(figsize=(7.0, 5.2), dpi=140)
    # Colors: low=blue, med=orange, high=red
    plt.step(t_lo, s_lo, where='post', color='#1f77b4', linewidth=2.0, label=f'Low (N={m_low.sum()}, events={int(event[m_low].sum())})')
    plt.step(t_md, s_md, where='post', color='#ff7f0e', linewidth=2.0, label=f'Medium (N={m_med.sum()}, events={int(event[m_med].sum())})')
    plt.step(t_hi, s_hi, where='post', color='#d62728', linewidth=2.0, label=f'High (N={m_high.sum()}, events={int(event[m_high].sum())})')

    if ci:
        for t, s, v, col in [(t_lo, s_lo, v_lo, '#1f77b4'), (t_md, s_md, v_md, '#ff7f0e'), (t_hi, s_hi, v_hi, '#d62728')]:
            se = np.sqrt(np.clip(v, 0.0, None))
            lo = np.clip(s - 1.96 * se, 0.0, 1.0); hi = np.clip(s + 1.96 * se, 0.0, 1.0)
            plt.fill_between(t, lo, hi, color=col, alpha=alpha_band, linewidth=0)

    plt.ylim(0.0, 1.0); plt.xlim(left=0.0)
    plt.grid(alpha=0.25, linestyle='--')
    plt.xlabel(xlab); plt.ylabel('Survival probability'); plt.title(title)
    plt.legend(loc='best', frameon=False); plt.tight_layout()
    try:
        plt.savefig(out_path)
    finally:
        plt.close()