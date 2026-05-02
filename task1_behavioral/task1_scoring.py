"""
Task 1: Behavioral Scoring Framework (Vectorized)
==================================================
PathAI Engine - Studor DS Screening Project

Fully vectorized implementation for speed. Computes 11 behavioral features
per student per week using pandas groupby operations instead of per-student loops.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'total_clicks', 'active_days', 'days_since_last', 'weekly_streak',
    'click_cv', 'activity_entropy', 'content_assess_ratio',
    'submission_timeliness', 'assess_completion',
    'click_trend_slope', 'wow_change'
]

ASSESSMENT_ACTIVITIES = {'quiz', 'externalquiz', 'questionnaire'}
CONTENT_ACTIVITIES = {'oucontent', 'resource', 'url', 'homepage', 'subpage',
                      'glossary', 'htmlactivity', 'page', 'folder'}


def load_oulad(data_dir: str) -> dict:
    """Load all OULAD CSV tables."""
    tables = {}
    for f in ['courses', 'vle', 'assessments', 'studentInfo', 'studentVle', 'studentAssessment', 'studentRegistration']:
        tables[f] = pd.read_csv(Path(data_dir) / f"{f}.csv")
        print(f"  {f}: {tables[f].shape}")
    return tables


def compute_weekly_features_vectorized(tables: dict, max_week: int = None) -> pd.DataFrame:
    """
    Vectorized computation of all 11 features per student per week.
    """
    svle = tables['studentVle'].copy()
    vle = tables['vle'][['id_site', 'code_module', 'code_presentation', 'activity_type']].copy()
    student_info = tables['studentInfo'][['code_module', 'code_presentation', 'id_student', 'final_result']].copy()
    assessments = tables['assessments'].copy()
    student_assess = tables['studentAssessment'].copy()
    courses = tables['courses'].copy()

    # Merge activity types
    svle = svle.merge(vle, on=['id_site', 'code_module', 'code_presentation'], how='left')
    svle['week'] = svle['date'] // 7 + 1

    # Max week per course
    courses['max_week'] = courses['module_presentation_length'] // 7
    if max_week:
        courses['max_week'] = courses['max_week'].clip(upper=max_week)
    
    # Filter clickstream to valid weeks
    svle = svle.merge(courses[['code_module', 'code_presentation', 'max_week']],
                      on=['code_module', 'code_presentation'], how='left')
    svle = svle[svle['week'] <= svle['max_week']]

    # Group key
    GK = ['code_module', 'code_presentation', 'id_student', 'week']

    print("  Computing volume features...")
    # Feature 1: total_clicks per student-week
    f_clicks = svle.groupby(GK)['sum_click'].sum().reset_index().rename(columns={'sum_click': 'total_clicks'})

    # Feature 2: active_days per student-week
    f_days = svle.groupby(GK)['date'].nunique().reset_index().rename(columns={'date': 'active_days'})

    # Feature 3: days_since_last (needs last active date per student up to each week)
    # We'll compute this differently: last active day within each week
    f_last_day = svle.groupby(GK)['date'].max().reset_index().rename(columns={'date': 'last_active_day'})

    print("  Computing diversity features...")
    # Feature 6: activity_entropy - compute via pivot + entropy
    type_clicks = svle.groupby(GK + ['activity_type'])['sum_click'].sum().reset_index()
    type_totals = type_clicks.groupby(GK)['sum_click'].transform('sum')
    type_clicks['prob'] = type_clicks['sum_click'] / type_totals
    type_clicks['neg_plogp'] = -type_clicks['prob'] * np.log(type_clicks['prob'] + 1e-10)
    f_entropy = type_clicks.groupby(GK)['neg_plogp'].sum().reset_index()
    f_entropy.columns = GK + ['activity_entropy']

    # Feature 7: content_assess_ratio
    svle['is_content'] = svle['activity_type'].isin(CONTENT_ACTIVITIES).astype(int)
    svle['is_assess'] = svle['activity_type'].isin(ASSESSMENT_ACTIVITIES).astype(int)
    
    content_clicks = svle[svle.is_content == 1].groupby(GK)['sum_click'].sum().reset_index().rename(columns={'sum_click': 'content_clicks'})
    assess_clicks = svle[svle.is_assess == 1].groupby(GK)['sum_click'].sum().reset_index().rename(columns={'sum_click': 'assess_clicks'})

    # Feature 5: click_cv - compute via aggregation instead of apply
    print("  Computing consistency features (CV)...")
    # Mean and std of daily clicks per student-week
    daily_clicks = svle.groupby(GK + ['date'])['sum_click'].sum().reset_index()
    daily_stats = daily_clicks.groupby(GK)['sum_click'].agg(['mean', 'std', 'count']).reset_index()
    daily_stats['click_cv'] = np.where(
        daily_stats['mean'] > 0,
        daily_stats['std'].fillna(0) / daily_stats['mean'],
        0
    )
    f_cv = daily_stats[GK + ['click_cv']]

    print("  Building full student-week grid...")
    # Create full grid of student x week
    student_weeks = []
    for (mod, pres), group in student_info.groupby(['code_module', 'code_presentation']):
        mw = courses[(courses.code_module == mod) & (courses.code_presentation == pres)]['max_week'].values
        if len(mw) == 0:
            continue
        mw = int(mw[0])
        weeks = pd.DataFrame({'week': range(1, mw + 1)})
        students = group[['code_module', 'code_presentation', 'id_student', 'final_result']].copy()
        students['_key'] = 1
        weeks['_key'] = 1
        cross = students.merge(weeks, on='_key').drop('_key', axis=1)
        student_weeks.append(cross)
    
    grid = pd.concat(student_weeks, ignore_index=True)
    print(f"  Grid size: {len(grid)} student-week rows")

    # Merge all features onto the grid
    grid = grid.merge(f_clicks, on=GK, how='left')
    grid = grid.merge(f_days, on=GK, how='left')
    grid = grid.merge(f_last_day, on=GK, how='left')
    grid = grid.merge(f_entropy, on=GK, how='left')
    grid = grid.merge(content_clicks, on=GK, how='left')
    grid = grid.merge(assess_clicks, on=GK, how='left')
    grid = grid.merge(f_cv, on=GK, how='left')

    # Fill NaN for weeks with no activity
    grid['total_clicks'] = grid['total_clicks'].fillna(0).astype(int)
    grid['active_days'] = grid['active_days'].fillna(0).astype(int)
    grid['activity_entropy'] = grid['activity_entropy'].fillna(0)
    grid['content_clicks'] = grid['content_clicks'].fillna(0)
    grid['assess_clicks'] = grid['assess_clicks'].fillna(0)
    grid['click_cv'] = grid['click_cv'].fillna(0)

    # Content vs assess ratio
    total_typed = grid['content_clicks'] + grid['assess_clicks']
    grid['content_assess_ratio'] = np.where(total_typed > 0, grid['content_clicks'] / total_typed, 0.5)
    grid.drop(['content_clicks', 'assess_clicks'], axis=1, inplace=True)

    print("  Computing temporal features (streak, recency, trend)...")
    # Sort for rolling computations
    grid = grid.sort_values(GK).reset_index(drop=True)
    SK = ['code_module', 'code_presentation', 'id_student']

    # Feature 3: days_since_last activity
    # Forward-fill last_active_day, then compute gap
    grid['last_active_day_ffill'] = grid.groupby(SK)['last_active_day'].ffill()
    grid['week_end_day'] = grid['week'] * 7 - 1
    grid['days_since_last'] = grid['week_end_day'] - grid['last_active_day_ffill']
    grid.loc[grid['last_active_day_ffill'].isna(), 'days_since_last'] = grid.loc[grid['last_active_day_ffill'].isna(), 'week_end_day']
    grid['days_since_last'] = grid['days_since_last'].clip(lower=0)
    grid.drop(['last_active_day', 'last_active_day_ffill', 'week_end_day'], axis=1, inplace=True)

    # Feature 4: weekly_streak
    grid['is_active'] = (grid['total_clicks'] > 0).astype(int)
    
    def compute_streak(series):
        streaks = []
        current = 0
        for val in series:
            if val > 0:
                current += 1
            else:
                current = 0
            streaks.append(current)
        return streaks
    
    grid['weekly_streak'] = grid.groupby(SK)['is_active'].transform(
        lambda x: pd.Series(compute_streak(x.values), index=x.index)
    )
    grid.drop('is_active', axis=1, inplace=True)

    # Feature 10: click_trend_slope (3-week rolling)
    def rolling_slope(series):
        result = []
        vals = series.values
        for i in range(len(vals)):
            if i < 2:
                result.append(0.0)
            else:
                window = vals[i-2:i+1]
                x = np.arange(3)
                slope = np.polyfit(x, window, 1)[0]
                result.append(slope)
        return result
    
    grid['click_trend_slope'] = grid.groupby(SK)['total_clicks'].transform(
        lambda x: pd.Series(rolling_slope(x), index=x.index)
    )

    # Feature 11: wow_change
    prev_clicks = grid.groupby(SK)['total_clicks'].shift(1)
    grid['wow_change'] = np.where(
        prev_clicks > 0,
        (grid['total_clicks'] - prev_clicks) / prev_clicks,
        0
    )
    grid['wow_change'] = grid['wow_change'].fillna(0).clip(-5, 5)  # Cap extreme values

    print("  Computing assessment features...")
    # Features 8 & 9: submission_timeliness and assess_completion
    # Pre-compute per student: which assessments were due by each week, and which were submitted
    assess_due = assessments[['code_module', 'code_presentation', 'id_assessment', 'date']].copy()
    assess_due['due_week'] = assess_due['date'] // 7 + 1
    assess_due.rename(columns={'date': 'due_date'}, inplace=True)

    # Merge with student submissions
    assess_sub = student_assess[['id_assessment', 'id_student', 'date_submitted']].merge(
        assess_due, on='id_assessment', how='right'
    )

    # For each student-week, count assessments due and submitted, compute timeliness
    grid['submission_timeliness'] = 0.0
    grid['assess_completion'] = 1.0

    # Build a lookup: for each (module, presentation, week), list of assessment ids due
    assess_due_by_week = {}
    for _, row in assess_due.iterrows():
        key = (row['code_module'], row['code_presentation'])
        if key not in assess_due_by_week:
            assess_due_by_week[key] = []
        assess_due_by_week[key].append((row['id_assessment'], row['due_date'], row['due_week']))

    # For each (module, presentation, student), get their submissions
    student_subs = assess_sub[assess_sub['id_student'].notna()].copy()
    student_subs['id_student'] = student_subs['id_student'].astype(int)
    sub_lookup = {}
    for _, row in student_subs.iterrows():
        if pd.isna(row['id_student']):
            continue
        key = (int(row['id_student']), int(row['id_assessment']))
        sub_lookup[key] = row['date_submitted']

    # Vectorized assessment features
    timeliness_vals = []
    completion_vals = []
    
    for _, row in grid.iterrows():
        mod, pres, sid, week = row['code_module'], row['code_presentation'], int(row['id_student']), int(row['week'])
        week_end_day = week * 7 - 1
        
        key = (mod, pres)
        if key not in assess_due_by_week:
            timeliness_vals.append(0.0)
            completion_vals.append(1.0)
            continue
        
        due_assessments = [(aid, dd) for aid, dd, dw in assess_due_by_week[key] if dd <= week_end_day]
        
        if not due_assessments:
            timeliness_vals.append(0.0)
            completion_vals.append(1.0)
            continue
        
        n_due = len(due_assessments)
        n_submitted = 0
        timeliness_sum = 0
        
        for aid, due_date in due_assessments:
            sub_key = (sid, aid)
            if sub_key in sub_lookup:
                n_submitted += 1
                timeliness_sum += (due_date - sub_lookup[sub_key])
        
        completion_vals.append(n_submitted / n_due)
        timeliness_vals.append(timeliness_sum / n_submitted if n_submitted > 0 else -30)
    
    grid['submission_timeliness'] = timeliness_vals
    grid['assess_completion'] = completion_vals

    print(f"  Done! Feature matrix shape: {grid.shape}")
    return grid


def train_engagement_model(features_df: pd.DataFrame) -> dict:
    """Train logistic regression to learn feature weights (Option D)."""
    df = features_df.copy()
    df['target'] = df['final_result'].isin(['Pass', 'Distinction']).astype(int)
    
    feat_data = df[FEATURE_COLS + ['target']].replace([np.inf, -np.inf], np.nan).dropna()
    X = feat_data[FEATURE_COLS].values
    y = feat_data['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    raw_weights = np.abs(model.coef_[0])
    normalized_weights = raw_weights / raw_weights.sum()
    feature_weights = dict(zip(FEATURE_COLS, normalized_weights))
    
    print("\n  Learned Feature Weights:")
    for feat, w in sorted(feature_weights.items(), key=lambda x: -x[1]):
        print(f"    {feat:30s}: {w:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_weights': feature_weights,
        'raw_coefficients': dict(zip(FEATURE_COLS, model.coef_[0]))
    }


def compute_engagement_scores(features_df: pd.DataFrame, model_info: dict) -> pd.DataFrame:
    """Compute engagement scores using Option D + F."""
    df = features_df.copy()
    weights = model_info['feature_weights']
    
    normalizer = MinMaxScaler()
    feat_values = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).values
    feat_normalized = normalizer.fit_transform(feat_values)
    
    # Invert features where lower = better
    dsl_idx = FEATURE_COLS.index('days_since_last')
    feat_normalized[:, dsl_idx] = 1 - feat_normalized[:, dsl_idx]
    cv_idx = FEATURE_COLS.index('click_cv')
    feat_normalized[:, cv_idx] = 1 - feat_normalized[:, cv_idx]
    
    weight_vector = np.array([weights[f] for f in FEATURE_COLS])
    raw_scores = feat_normalized @ weight_vector
    
    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max > score_min:
        df['absolute_score'] = ((raw_scores - score_min) / (score_max - score_min) * 100).round(1)
    else:
        df['absolute_score'] = 50.0
    
    # Peer percentile within course-week
    df['peer_percentile'] = df.groupby(
        ['code_module', 'code_presentation', 'week']
    )['absolute_score'].rank(pct=True).mul(100).round(1)
    
    # Combined: 60% absolute + 40% peer
    df['engagement_score'] = (0.6 * df['absolute_score'] + 0.4 * df['peer_percentile']).round(1)
    
    # Velocity and acceleration
    SK = ['code_module', 'code_presentation', 'id_student']
    df = df.sort_values(SK + ['week'])
    df['score_velocity'] = df.groupby(SK)['engagement_score'].diff().fillna(0).round(2)
    df['score_acceleration'] = df.groupby(SK)['score_velocity'].diff().fillna(0).round(2)
    
    # Normalized week (secondary view)
    max_weeks = df.groupby(['code_module', 'code_presentation'])['week'].max().reset_index()
    max_weeks.columns = ['code_module', 'code_presentation', 'total_weeks']
    df = df.merge(max_weeks, on=['code_module', 'code_presentation'], how='left')
    df['pct_complete'] = (df['week'] / df['total_weeks'] * 100).round(1)
    
    return df


def build_trajectory_matrix(scored_df, n_weeks=20):
    """Build fixed-length trajectory matrix for clustering."""
    SK = ['code_module', 'code_presentation', 'id_student']
    
    stu_counts = scored_df.groupby(SK)['week'].count().reset_index()
    stu_counts = stu_counts[stu_counts.week >= 3]
    
    valid = scored_df.merge(stu_counts[SK], on=SK, how='inner')
    
    trajectories = []
    student_keys = []
    outcomes = []
    
    for key, group in valid.groupby(SK):
        scores = group.sort_values('week')['engagement_score'].values
        outcome = group['final_result'].iloc[0]
        
        if len(scores) >= n_weeks:
            traj = scores[:n_weeks]
        else:
            traj = np.pad(scores, (0, n_weeks - len(scores)), mode='constant', constant_values=0)
        
        trajectories.append(traj)
        student_keys.append(key)
        outcomes.append(outcome)
    
    return np.array(trajectories), student_keys, outcomes


def discover_archetypes_kmeans(traj_matrix, outcomes, k_range=(3, 7)):
    """K-Means clustering on trajectories."""
    print("\n  K-Means Archetype Discovery:")
    best_k, best_sil = 3, -1
    results = {}
    
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(traj_matrix)
        sil = silhouette_score(traj_matrix, labels, sample_size=min(5000, len(labels)))
        results[k] = {'model': km, 'labels': labels, 'silhouette': sil}
        print(f"    k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    print(f"    Best k: {best_k}")
    best = results[best_k]
    labels = best['labels']
    outcome_arr = np.array(outcomes)
    
    archetypes = {}
    for c in range(best_k):
        mask = labels == c
        cluster_trajs = traj_matrix[mask]
        cluster_outcomes = outcome_arr[mask]
        
        mean_traj = cluster_trajs.mean(axis=0)
        unique, counts = np.unique(cluster_outcomes, return_counts=True)
        outcome_dist = dict(zip(unique, (counts / counts.sum() * 100).round(1)))
        overall_slope = np.polyfit(np.arange(len(mean_traj)), mean_traj, 1)[0]
        
        first_half = mean_traj[:len(mean_traj)//2].mean()
        label = _classify_archetype(mean_traj, overall_slope, first_half, c)
        
        archetypes[c] = {
            'label': label, 'mean_trajectory': mean_traj,
            'n_students': int(mask.sum()), 'outcome_distribution': outcome_dist,
            'overall_slope': round(overall_slope, 3), 'mean_score': round(mean_traj.mean(), 1)
        }
        print(f"    Cluster {c} - {label}: n={mask.sum()}, mean={mean_traj.mean():.1f}, slope={overall_slope:.2f}")
        print(f"      Outcomes: {outcome_dist}")
    
    return {'method': 'kmeans', 'k': best_k, 'labels': labels,
            'archetypes': archetypes, 'silhouette': best_sil}


def discover_archetypes_dtw(traj_matrix, outcomes, k_range=(3, 7), sample_size=1500):
    """DTW-based hierarchical clustering."""
    try:
        from dtaidistance import dtw
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        print("  DTW not available, skipping.")
        return None
    
    print("\n  DTW Archetype Discovery:")
    
    n = len(traj_matrix)
    if n > sample_size:
        idx = np.random.RandomState(42).choice(n, sample_size, replace=False)
        sampled = traj_matrix[idx]
        sampled_outcomes = [outcomes[i] for i in idx]
    else:
        sampled = traj_matrix
        sampled_outcomes = outcomes
        idx = np.arange(n)
    
    print(f"    Computing DTW distances for {len(sampled)} trajectories...")
    series = [row.astype(np.double) for row in sampled]
    
    try:
        dm = dtw.distance_matrix_fast(series)
    except:
        dm = dtw.distance_matrix(series)
    
    dm = np.nan_to_num(dm, nan=1e6, posinf=1e6)
    np.fill_diagonal(dm, 0)
    dm = (dm + dm.T) / 2
    
    condensed = squareform(dm)
    Z = linkage(condensed, method='ward')
    
    best_k, best_sil = 4, -1
    for k in range(k_range[0], k_range[1] + 1):
        labels = fcluster(Z, t=k, criterion='maxclust')
        try:
            sil = silhouette_score(dm, labels, metric='precomputed')
            print(f"    k={k}: silhouette={sil:.4f}")
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except:
            pass
    
    labels = fcluster(Z, t=best_k, criterion='maxclust')
    print(f"    Best k: {best_k}")
    
    outcome_arr = np.array(sampled_outcomes)
    archetypes = {}
    for c in sorted(np.unique(labels)):
        mask = labels == c
        cluster_trajs = sampled[mask]
        mean_traj = cluster_trajs.mean(axis=0)
        cluster_outcomes = outcome_arr[mask]
        unique, counts = np.unique(cluster_outcomes, return_counts=True)
        outcome_dist = dict(zip(unique, (counts / counts.sum() * 100).round(1)))
        overall_slope = np.polyfit(np.arange(len(mean_traj)), mean_traj, 1)[0]
        first_half = mean_traj[:len(mean_traj)//2].mean()
        label = _classify_archetype(mean_traj, overall_slope, first_half, c)
        
        archetypes[int(c)] = {
            'label': label, 'mean_trajectory': mean_traj,
            'n_students': int(mask.sum()), 'outcome_distribution': outcome_dist,
            'overall_slope': round(overall_slope, 3), 'mean_score': round(mean_traj.mean(), 1)
        }
        print(f"    Cluster {c} - {label}: n={mask.sum()}, mean={mean_traj.mean():.1f}")
    
    return {'method': 'dtw', 'k': best_k, 'labels': labels,
            'archetypes': archetypes, 'silhouette': best_sil, 'sample_indices': idx}


def _classify_archetype(mean_traj, slope, first_half, c):
    """Assign a human-readable label to a cluster based on trajectory shape."""
    if slope > 1 and first_half < 40:
        return "Late Recoverer"
    elif slope < -1 and first_half > 40:
        return "Fading Away"
    elif mean_traj.mean() > 65:
        return "Steady Engager"
    elif mean_traj.mean() < 25:
        return "Minimal"
    elif np.std(mean_traj) > 15:
        return "Erratic / Crammer"
    elif slope < -0.5:
        return "Early Dropout"
    else:
        return f"Moderate (Cluster {c})"


def shap_importance_by_week(features_df, model_info, weeks=None):
    """SHAP feature importance per week."""
    import shap
    
    df = features_df.copy()
    df['target'] = df['final_result'].isin(['Pass', 'Distinction']).astype(int)
    
    if weeks is None:
        weeks = sorted(df['week'].unique())[:15]
    
    scaler = model_info['scaler']
    model = model_info['model']
    weekly_importance = {}
    
    print("\n  SHAP importance by week:")
    for w in weeks:
        wk = df[df.week == w][FEATURE_COLS + ['target']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(wk) < 50:
            continue
        X = scaler.transform(wk[FEATURE_COLS].values)
        explainer = shap.LinearExplainer(model, X)
        sv = explainer.shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        total = mean_abs.sum()
        importance = {k: round(v/total, 4) for k, v in zip(FEATURE_COLS, mean_abs)}
        weekly_importance[w] = importance
        top3 = sorted(importance.items(), key=lambda x: -x[1])[:3]
        print(f"    Week {w:2d}: {top3[0][0]}({top3[0][1]:.3f}), {top3[1][0]}({top3[1][1]:.3f}), {top3[2][0]}({top3[2][1]:.3f})")
    
    return weekly_importance


def compute_feature_rationale(features_df):
    """Statistics by outcome group for feature justification."""
    df = features_df.copy()
    df['outcome_group'] = df['final_result'].map({
        'Distinction': 'Success', 'Pass': 'Success',
        'Fail': 'At-Risk', 'Withdrawn': 'At-Risk'
    })
    
    rows = []
    for feat in FEATURE_COLS:
        for group in ['Success', 'At-Risk']:
            vals = df[df.outcome_group == group][feat].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({'feature': feat, 'group': group,
                        'mean': vals.mean(), 'median': vals.median(), 'std': vals.std(), 'n': len(vals)})
    
    rationale = pd.DataFrame(rows)
    
    print("\n  Feature Rationale (Mean by group):")
    for feat in FEATURE_COLS:
        s = rationale[(rationale.feature == feat) & (rationale.group == 'Success')]['mean'].values[0]
        a = rationale[(rationale.feature == feat) & (rationale.group == 'At-Risk')]['mean'].values[0]
        diff = ((s - a) / max(abs(a), 0.001)) * 100
        print(f"    {feat:30s}: Success={s:8.2f}  At-Risk={a:8.2f}  Diff={diff:+.1f}%")
    
    return rationale


def run_task1(data_dir, output_dir=None, max_week=None, sample_courses=None):
    """Full Task 1 pipeline."""
    if output_dir is None:
        output_dir = Path(data_dir) / "task1_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TASK 1: BEHAVIORAL SCORING FRAMEWORK")
    print("=" * 60)
    
    print("\n[1/7] Loading data...")
    tables = load_oulad(data_dir)
    
    if sample_courses:
        courses = tables['courses'].head(sample_courses)
        tables['courses'] = courses
        for key in ['studentInfo', 'studentVle', 'studentRegistration', 'assessments', 'vle']:
            if 'code_module' in tables[key].columns:
                tables[key] = tables[key].merge(
                    courses[['code_module', 'code_presentation']],
                    on=['code_module', 'code_presentation'], how='inner')
        valid_ids = tables['assessments']['id_assessment'].unique()
        tables['studentAssessment'] = tables['studentAssessment'][
            tables['studentAssessment']['id_assessment'].isin(valid_ids)]
        print(f"  Sampled to {sample_courses} course-presentations")
    
    print("\n[2/7] Engineering features (vectorized)...")
    features_df = compute_weekly_features_vectorized(tables, max_week=max_week)
    features_df.to_csv(output_dir / "weekly_features.csv", index=False)
    
    print("\n[3/7] Feature rationale...")
    rationale = compute_feature_rationale(features_df)
    rationale.to_csv(output_dir / "feature_rationale.csv", index=False)
    
    print("\n[4/7] Training engagement model...")
    model_info = train_engagement_model(features_df)
    
    print("\n[5/7] Computing engagement scores...")
    scored_df = compute_engagement_scores(features_df, model_info)
    scored_df.to_csv(output_dir / "engagement_scores.csv", index=False)
    
    print("\n[6/7] Discovering archetypes...")
    traj_matrix, student_keys, outcomes = build_trajectory_matrix(scored_df, n_weeks=20)
    print(f"  Trajectory matrix: {traj_matrix.shape}")
    
    kmeans_result = discover_archetypes_kmeans(traj_matrix, outcomes)
    dtw_result = discover_archetypes_dtw(traj_matrix, outcomes, sample_size=1500)
    
    print("\n[7/7] SHAP analysis...")
    shap_imp = shap_importance_by_week(features_df, model_info)
    
    results = {
        'features_df': features_df, 'scored_df': scored_df,
        'model_info': model_info, 'rationale_df': rationale,
        'traj_matrix': traj_matrix, 'student_keys': student_keys,
        'outcomes': outcomes, 'kmeans_archetypes': kmeans_result,
        'dtw_archetypes': dtw_result, 'shap_importance': shap_imp,
        'output_dir': output_dir
    }
    
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETE")
    print("=" * 60)
    return results


import argparse
from pathlib import Path

# Connect your visualization script
from task1_visualizations import generate_all_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1 Engagement Scoring")
    
    # BULLETPROOF PATHS: This ensures it works regardless of how you run it in VS Code
    script_dir = Path(__file__).resolve().parent  # Gets the exact path of the task1_behavioral folder
    project_root = script_dir.parent              # Goes up exactly one level to the main project folder
    
    default_data_dir = project_root / 'data'
    default_output_dir = project_root / 'output' / 'task1_output'
    
    parser.add_argument('--data_dir', type=str, default=str(default_data_dir), help='Directory containing OULAD CSV files')
    parser.add_argument('--output_dir', type=str, default=str(default_output_dir), help='Directory to save outputs')
    
    args = parser.parse_args()

    # Run the main pipeline
    results = run_task1(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_week=20
    )

    # Generate the plots and save them
    generate_all_plots(results)