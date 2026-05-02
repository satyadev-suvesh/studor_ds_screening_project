"""
Task 2: Predictive Disengagement Model
=======================================
PathAI Engine - Studor DS Screening Project

Predicts which students will withdraw or fail before Week 6 using only
data available at the prediction point. Includes:

- Leakage-proof feature construction (Week <= 6 only)
- Hybrid feature aggregation (week-6 snapshot + cross-week stats)
- Model comparison: Logistic Regression, Random Forest, LightGBM
- Threshold tuning optimized for Recall
- Probability calibration with reliability diagrams
- Leave-one-course-out cross-validation
- SHAP explainability (global + per-student)
- Temporal stability analysis (week 3 vs week 6)
- Fairness audit across demographic groups
- Cost-benefit threshold analysis
- What-if counterfactual analysis
- Staff notification design with mock Streamlit dashboard

Author: Satyadev
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
from scipy.stats import entropy as scipy_entropy
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

PREDICTION_WEEK = 6
EARLY_WEEK = 3  # for temporal stability analysis

ASSESSMENT_ACTIVITIES = {'quiz', 'externalquiz', 'questionnaire'}
CONTENT_ACTIVITIES = {
    'oucontent', 'resource', 'url', 'homepage', 'subpage',
    'glossary', 'htmlactivity', 'page', 'folder'
}

COLORS = {
    'Distinction': '#2ecc71', 'Pass': '#3498db',
    'Fail': '#e74c3c', 'Withdrawn': '#95a5a6',
    'Success': '#2ecc71', 'At-Risk': '#e74c3c'
}

# ---------------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------------

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
default_data_dir = project_root / 'data'
default_output_dir = project_root / 'output' / 'task2_output'


# ===================================================================
# 1. DATA LOADING & WEEKLY FEATURE COMPUTATION
# ===================================================================

def load_oulad(data_dir: Path) -> dict:
    """Load all OULAD CSV tables."""
    tables = {}
    for f in ['courses', 'vle', 'assessments', 'studentInfo',
              'studentVle', 'studentAssessment', 'studentRegistration']:
        tables[f] = pd.read_csv(data_dir / f"{f}.csv")
        print(f"  {f}: {tables[f].shape}")
    return tables


def compute_weekly_features(tables: dict, max_week: int) -> pd.DataFrame:
    """
    Vectorized computation of 11 behavioral features per student per week,
    capped at max_week. Mirrors the Task 1 pipeline exactly so that we
    reuse the same feature definitions.
    """
    svle = tables['studentVle'].copy()
    vle = tables['vle'][['id_site', 'code_module', 'code_presentation',
                          'activity_type']].copy()
    student_info = tables['studentInfo'][
        ['code_module', 'code_presentation', 'id_student', 'final_result']
    ].copy()
    assessments = tables['assessments'].copy()
    student_assess = tables['studentAssessment'].copy()
    courses = tables['courses'].copy()

    # Merge activity types into clickstream
    svle = svle.merge(vle, on=['id_site', 'code_module', 'code_presentation'],
                      how='left')
    svle['week'] = svle['date'] // 7 + 1
    svle = svle[svle['week'] <= max_week]

    GK = ['code_module', 'code_presentation', 'id_student', 'week']

    # --- Volume ---
    f_clicks = (svle.groupby(GK)['sum_click'].sum()
                .reset_index().rename(columns={'sum_click': 'total_clicks'}))
    f_days = (svle.groupby(GK)['date'].nunique()
              .reset_index().rename(columns={'date': 'active_days'}))
    f_last_day = (svle.groupby(GK)['date'].max()
                  .reset_index().rename(columns={'date': 'last_active_day'}))

    # --- Entropy ---
    tc = svle.groupby(GK + ['activity_type'])['sum_click'].sum().reset_index()
    tt = tc.groupby(GK)['sum_click'].transform('sum')
    tc['prob'] = tc['sum_click'] / tt
    tc['neg_plogp'] = -tc['prob'] * np.log(tc['prob'] + 1e-10)
    f_entropy = (tc.groupby(GK)['neg_plogp'].sum()
                 .reset_index().rename(columns={'neg_plogp': 'activity_entropy'}))

    # --- Content vs assess ratio ---
    svle['is_content'] = svle['activity_type'].isin(CONTENT_ACTIVITIES).astype(int)
    svle['is_assess'] = svle['activity_type'].isin(ASSESSMENT_ACTIVITIES).astype(int)
    content_c = (svle[svle.is_content == 1].groupby(GK)['sum_click'].sum()
                 .reset_index().rename(columns={'sum_click': 'content_clicks'}))
    assess_c = (svle[svle.is_assess == 1].groupby(GK)['sum_click'].sum()
                .reset_index().rename(columns={'sum_click': 'assess_clicks'}))

    # --- CV ---
    daily = svle.groupby(GK + ['date'])['sum_click'].sum().reset_index()
    ds = daily.groupby(GK)['sum_click'].agg(['mean', 'std']).reset_index()
    ds['click_cv'] = np.where(ds['mean'] > 0, ds['std'].fillna(0) / ds['mean'], 0)
    f_cv = ds[GK + ['click_cv']]

    # --- Build full grid ---
    student_weeks = []
    for (mod, pres), grp in student_info.groupby(
            ['code_module', 'code_presentation']):
        weeks = pd.DataFrame({'week': range(1, max_week + 1)})
        stu = grp[['code_module', 'code_presentation',
                    'id_student', 'final_result']].copy()
        stu['_k'] = 1; weeks['_k'] = 1
        student_weeks.append(stu.merge(weeks, on='_k').drop('_k', axis=1))
    grid = pd.concat(student_weeks, ignore_index=True)

    # Merge features
    for df in [f_clicks, f_days, f_last_day, f_entropy,
               content_c, assess_c, f_cv]:
        grid = grid.merge(df, on=GK, how='left')

    grid['total_clicks'] = grid['total_clicks'].fillna(0).astype(int)
    grid['active_days'] = grid['active_days'].fillna(0).astype(int)
    grid['activity_entropy'] = grid['activity_entropy'].fillna(0)
    grid['content_clicks'] = grid['content_clicks'].fillna(0)
    grid['assess_clicks'] = grid['assess_clicks'].fillna(0)
    grid['click_cv'] = grid['click_cv'].fillna(0)

    total_typed = grid['content_clicks'] + grid['assess_clicks']
    grid['content_assess_ratio'] = np.where(
        total_typed > 0, grid['content_clicks'] / total_typed, 0.5)
    grid.drop(['content_clicks', 'assess_clicks'], axis=1, inplace=True)

    # --- Temporal features ---
    SK = ['code_module', 'code_presentation', 'id_student']
    grid = grid.sort_values(SK + ['week']).reset_index(drop=True)

    grid['last_active_day_ff'] = grid.groupby(SK)['last_active_day'].ffill()
    grid['week_end_day'] = grid['week'] * 7 - 1
    grid['days_since_last'] = (grid['week_end_day']
                                - grid['last_active_day_ff'].fillna(-999))
    grid.loc[grid['last_active_day_ff'].isna(), 'days_since_last'] = \
        grid.loc[grid['last_active_day_ff'].isna(), 'week_end_day']
    grid['days_since_last'] = grid['days_since_last'].clip(lower=0)
    grid.drop(['last_active_day', 'last_active_day_ff', 'week_end_day'],
              axis=1, inplace=True)

    # Streak
    def _streak(s):
        out, cur = [], 0
        for v in s:
            cur = cur + 1 if v > 0 else 0
            out.append(cur)
        return out

    grid['weekly_streak'] = grid.groupby(SK)['total_clicks'].transform(
        lambda x: pd.Series(_streak(x.values), index=x.index))

    # Trend slope (3-week rolling)
    def _slope(s):
        v = s.values
        out = [0.0, 0.0]
        for i in range(2, len(v)):
            out.append(np.polyfit(np.arange(3), v[i-2:i+1], 1)[0])
        return out

    grid['click_trend_slope'] = grid.groupby(SK)['total_clicks'].transform(
        lambda x: pd.Series(_slope(x), index=x.index))

    # WoW change
    prev = grid.groupby(SK)['total_clicks'].shift(1)
    grid['wow_change'] = np.where(prev > 0,
                                   (grid['total_clicks'] - prev) / prev, 0)
    grid['wow_change'] = grid['wow_change'].fillna(0).clip(-5, 5)

    # --- Assessment features ---
    assess_due = assessments[['code_module', 'code_presentation',
                               'id_assessment', 'date']].copy()
    assess_due.rename(columns={'date': 'due_date'}, inplace=True)

    sub_lookup = {}
    merged = student_assess[['id_assessment', 'id_student',
                              'date_submitted']].merge(
        assess_due, on='id_assessment', how='inner')
    for _, r in merged.iterrows():
        if pd.notna(r['id_student']):
            sub_lookup[(int(r['id_student']), int(r['id_assessment']))] = \
                r['date_submitted']

    assess_by_course = {}
    for _, r in assess_due.iterrows():
        key = (r['code_module'], r['code_presentation'])
        assess_by_course.setdefault(key, []).append(
            (int(r['id_assessment']), r['due_date']))

    timeliness_vals, completion_vals = [], []
    for _, row in grid.iterrows():
        mod, pres = row['code_module'], row['code_presentation']
        sid, week = int(row['id_student']), int(row['week'])
        wed = week * 7 - 1

        due = [(a, d) for a, d in assess_by_course.get((mod, pres), [])
               if d <= wed]
        if not due:
            timeliness_vals.append(0.0)
            completion_vals.append(1.0)
            continue

        n_sub, t_sum = 0, 0.0
        for aid, dd in due:
            sd = sub_lookup.get((sid, aid))
            if sd is not None:
                n_sub += 1
                t_sum += (dd - sd)

        completion_vals.append(n_sub / len(due))
        timeliness_vals.append(t_sum / n_sub if n_sub > 0 else -30)

    grid['submission_timeliness'] = timeliness_vals
    grid['assess_completion'] = completion_vals

    return grid


# ===================================================================
# 2. PREDICTION FEATURE MATRIX (HYBRID AGGREGATION - Option D)
# ===================================================================

BASE_FEATURES = [
    'total_clicks', 'active_days', 'days_since_last', 'weekly_streak',
    'click_cv', 'activity_entropy', 'content_assess_ratio',
    'submission_timeliness', 'assess_completion',
    'click_trend_slope', 'wow_change'
]


def build_prediction_features(weekly_df: pd.DataFrame,
                               tables: dict,
                               cutoff_week: int) -> pd.DataFrame:
    """
    Build one feature vector per student using data up to cutoff_week.

    Hybrid aggregation (Option D):
      - Week-N snapshot (last available week values)
      - Cross-week aggregates (mean, std, slope)
      - Prediction-specific early warning signals
      - Educational demographic features
    """
    wk = weekly_df[weekly_df['week'] <= cutoff_week].copy()
    SK = ['code_module', 'code_presentation', 'id_student']

    # --- Snapshot features (week = cutoff_week) ---
    snap = wk[wk['week'] == cutoff_week][SK + BASE_FEATURES].copy()
    snap.columns = SK + [f'{f}_last' for f in BASE_FEATURES]

    # --- Aggregate features across weeks 1..cutoff ---
    agg_dict = {f: ['mean', 'std'] for f in BASE_FEATURES}
    agg = wk.groupby(SK).agg(agg_dict).reset_index()
    agg.columns = SK + [f'{f}_{stat}' for f in BASE_FEATURES
                        for stat in ['mean', 'std']]

    # --- Slope of each feature over weeks ---
    def _compute_slopes(group):
        x = np.arange(len(group))
        slopes = {}
        for f in BASE_FEATURES:
            vals = group[f].values
            if len(vals) >= 2:
                slopes[f'{f}_slope'] = np.polyfit(x, vals, 1)[0]
            else:
                slopes[f'{f}_slope'] = 0.0
        return pd.Series(slopes)

    slopes = wk.groupby(SK).apply(_compute_slopes,
                                   include_groups=False).reset_index()

    # --- Early warning signals ---
    ew = wk.groupby(SK).agg(
        inactive_weeks=('total_clicks', lambda x: (x == 0).sum()),
        peak_clicks=('total_clicks', 'max'),
        last_clicks=('total_clicks', 'last'),
    ).reset_index()
    ew['peak_to_current_ratio'] = np.where(
        ew['peak_clicks'] > 0, ew['last_clicks'] / ew['peak_clicks'], 0)

    # Assessment gap trend (is timeliness getting worse?)
    assess_trend = wk.groupby(SK).apply(
        lambda g: np.polyfit(np.arange(len(g)),
                             g['submission_timeliness'].values, 1)[0]
        if len(g) >= 2 else 0.0, include_groups=False
    ).reset_index().rename(columns={0: 'assess_gap_trend'})

    # --- Merge all ---
    features = snap.merge(agg, on=SK, how='outer')
    features = features.merge(slopes, on=SK, how='left')
    features = features.merge(ew[SK + ['inactive_weeks',
                                        'peak_to_current_ratio']],
                               on=SK, how='left')
    features = features.merge(assess_trend, on=SK, how='left')

    # --- Add outcome label ---
    info = tables['studentInfo'][SK + ['final_result']].drop_duplicates()
    features = features.merge(info, on=SK, how='left')
    features['target'] = features['final_result'].isin(
        ['Withdrawn', 'Fail']).astype(int)

    # --- Educational demographics ---
    demo_cols = ['gender', 'region', 'highest_education', 'imd_band',
                 'age_band', 'num_of_prev_attempts', 'studied_credits',
                 'disability']
    demo = tables['studentInfo'][SK + demo_cols].drop_duplicates()
    features = features.merge(demo, on=SK, how='left')

    # Encode categoricals
    cat_cols = ['gender', 'region', 'highest_education', 'imd_band',
                'age_band', 'disability']
    for col in cat_cols:
        if col in features.columns:
            features[col] = LabelEncoder().fit_transform(
                features[col].astype(str))

    features = features.fillna(0)
    return features


# ===================================================================
# 3. MODEL TRAINING & EVALUATION
# ===================================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all feature columns (everything except metadata and target)."""
    exclude = {'code_module', 'code_presentation', 'id_student',
               'final_result', 'target'}
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate_models(features_df: pd.DataFrame,
                               output_dir: Path) -> dict:
    """
    Train Logistic Regression, Random Forest, and LightGBM.
    Use leave-one-course-out CV. Compare and select the best model.
    Then tune threshold for recall.
    """
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].values
    y = features_df['target'].values
    courses = (features_df['code_module'] + '_'
               + features_df['code_presentation']).values

    unique_courses = np.unique(courses)
    print(f"\n  Leave-one-course-out CV with {len(unique_courses)} courses")

    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1),
    }
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1)

    results = {}
    all_preds = {name: {'y_true': [], 'y_prob': [], 'y_pred': []}
                 for name in models}

    for held_out in unique_courses:
        train_mask = courses != held_out
        test_mask = courses == held_out

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in models.items():
            m = _clone_model(model)
            if name == 'LogisticRegression':
                m.fit(X_train_s, y_train)
                probs = m.predict_proba(X_test_s)[:, 1]
            else:
                m.fit(X_train, y_train)
                probs = m.predict_proba(X_test)[:, 1]

            all_preds[name]['y_true'].extend(y_test.tolist())
            all_preds[name]['y_prob'].extend(probs.tolist())

    # Evaluate each model
    print("\n  Model Comparison (Leave-One-Course-Out):")
    print(f"  {'Model':25s} {'AUC':>8s} {'F1':>8s} {'Recall':>8s} {'Prec':>8s}")
    print("  " + "-" * 55)

    for name in models:
        yt = np.array(all_preds[name]['y_true'])
        yp = np.array(all_preds[name]['y_prob'])
        yhat = (yp >= 0.5).astype(int)

        auc = roc_auc_score(yt, yp)
        f1 = f1_score(yt, yhat)
        rec = recall_score(yt, yhat)
        prec = precision_score(yt, yhat)

        results[name] = {
            'auc': auc, 'f1': f1, 'recall': rec, 'precision': prec,
            'y_true': yt, 'y_prob': yp
        }
        print(f"  {name:25s} {auc:8.4f} {f1:8.4f} {rec:8.4f} {prec:8.4f}")

    # Select best model by AUC
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\n  Best model: {best_name} (AUC = {results[best_name]['auc']:.4f})")

    # --- Threshold tuning for recall ---
    yt = results[best_name]['y_true']
    yp = results[best_name]['y_prob']
    best_thresh, best_f1, best_rec = _tune_threshold(yt, yp, min_recall=0.80)
    print(f"  Tuned threshold: {best_thresh:.3f} "
          f"(Recall={best_rec:.3f}, F1={best_f1:.3f})")

    yhat_tuned = (yp >= best_thresh).astype(int)

    # --- Retrain best model on full data ---
    print(f"\n  Retraining {best_name} on full dataset...")
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    final_model = _clone_model(models[best_name])

    if best_name == 'LogisticRegression':
        final_model.fit(X_full_s, y)
    else:
        final_model.fit(X, y)

    # Calibrate
    print("  Calibrating probabilities (isotonic)...")
    cal_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
    if best_name == 'LogisticRegression':
        cal_model.fit(X_full_s, y)
    else:
        cal_model.fit(X, y)

    return {
        'best_model_name': best_name,
        'final_model': final_model,
        'calibrated_model': cal_model,
        'scaler': scaler_full,
        'feature_cols': feat_cols,
        'threshold': best_thresh,
        'cv_results': results,
        'all_preds': all_preds,
        'features_df': features_df
    }


def _clone_model(model):
    """Create a fresh copy of a model with the same hyperparameters."""
    from sklearn.base import clone
    return clone(model)


def _tune_threshold(y_true, y_prob, min_recall=0.80):
    """Sweep thresholds to find the one maximizing F1 at >= min_recall."""
    best_t, best_f1, best_rec = 0.5, 0, 0
    for t in np.arange(0.10, 0.90, 0.01):
        yhat = (y_prob >= t).astype(int)
        rec = recall_score(y_true, yhat)
        f1 = f1_score(y_true, yhat)
        if rec >= min_recall and f1 > best_f1:
            best_t, best_f1, best_rec = t, f1, rec
    return best_t, best_f1, best_rec


# ===================================================================
# 4. CALIBRATION ANALYSIS
# ===================================================================

def calibration_analysis(model_results: dict, output_dir: Path):
    """
    Reliability diagram + calibration table for administrators.
    """
    best = model_results['best_model_name']
    yt = model_results['cv_results'][best]['y_true']
    yp = model_results['cv_results'][best]['y_prob']

    # Reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy='uniform')
    ax.plot(mean_pred, frac_pos, 's-', color='#e74c3c', linewidth=2,
            label='Model')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=13,
                 fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Calibration table (administrator-friendly)
    ax = axes[1]
    ax.axis('off')

    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%"
                  for i in range(len(bins)-1)]

    digitized = np.digitize(yp, bins) - 1
    table_data = []
    for i in range(10):
        mask = digitized == i
        n = mask.sum()
        if n > 0:
            actual_rate = yt[mask].mean() * 100
        else:
            actual_rate = 0
        table_data.append([bin_labels[i], str(n), f"{actual_rate:.1f}%"])

    table = ax.table(cellText=table_data,
                     colLabels=['Risk Band', 'Students', 'Actual Withdrawal Rate'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title('Calibration Table for Advisors', fontsize=13,
                 fontweight='bold', pad=20)

    brier = brier_score_loss(yt, yp)
    fig.suptitle(f'Calibration Analysis (Brier Score = {brier:.4f})',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'calibration_analysis.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved calibration_analysis.png")

    return {'brier_score': brier, 'calibration_table': table_data}


# ===================================================================
# 5. SHAP EXPLAINABILITY
# ===================================================================

def shap_analysis(model_results: dict, output_dir: Path):
    """Global SHAP + dependence plots + force plots for sample students."""
    feat_cols = model_results['feature_cols']
    df = model_results['features_df']
    X = df[feat_cols].values
    model = model_results['final_model']
    best_name = model_results['best_model_name']

    if best_name == 'LogisticRegression':
        X_input = model_results['scaler'].transform(X)
        explainer = shap.LinearExplainer(model, X_input)
        sv = explainer.shap_values(X_input)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # Handle different SHAP output formats
        if isinstance(sv, list):
            sv = sv[1]  # class 1 (at-risk)
        elif sv.ndim == 3:
            sv = sv[:, :, 1]  # (samples, features, classes) -> class 1

    # Global feature importance
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance = sorted(zip(feat_cols, mean_abs_shap),
                        key=lambda x: -x[1])

    print("\n  Top 10 Features (SHAP):")
    for feat, val in importance[:10]:
        print(f"    {feat:40s}: {val:.4f}")

    # Plot: global importance (top 15)
    top_n = 15
    fig, ax = plt.subplots(figsize=(10, 7))
    top_feats = importance[:top_n]
    names = [f[0].replace('_', ' ').title() for f in top_feats][::-1]
    vals = [f[1] for f in top_feats][::-1]
    bars = ax.barh(names, vals, color='#3498db', edgecolor='white')
    for bar in bars[-3:]:
        bar.set_color('#e74c3c')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Top Features Driving Disengagement Risk',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'shap_global_importance.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved shap_global_importance.png")

    # Dependence plots for top 3
    top3_feats = [importance[i][0] for i in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat in enumerate(top3_feats):
        idx = feat_cols.index(feat)
        ax = axes[i]
        x_vals = X[:, idx] if best_name != 'LogisticRegression' \
            else X_input[:, idx]
        ax.scatter(x_vals, sv[:, idx], alpha=0.3, s=10,
                  c=df['target'].values, cmap='RdYlGn_r')
        ax.set_xlabel(feat.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('SHAP Value', fontsize=11)
        ax.set_title(f'Dependence: {feat.replace("_", " ").title()}',
                    fontsize=12, fontweight='bold')
        ax.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / 'shap_dependence_top3.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved shap_dependence_top3.png")

    return {
        'shap_values': sv,
        'feature_importance': importance,
        'top3_features': top3_feats,
        'explainer': explainer
    }


# ===================================================================
# 6. CONFUSION MATRIX & METRICS PLOTS
# ===================================================================

def plot_confusion_and_metrics(model_results: dict, output_dir: Path):
    """Confusion matrix, ROC, Precision-Recall curve, threshold analysis."""
    best = model_results['best_model_name']
    yt = model_results['cv_results'][best]['y_true']
    yp = model_results['cv_results'][best]['y_prob']
    thresh = model_results['threshold']
    yhat = (yp >= thresh).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Confusion matrix
    ax = axes[0, 0]
    cm = confusion_matrix(yt, yhat)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pass/Distinction', 'Withdraw/Fail'],
                yticklabels=['Pass/Distinction', 'Withdraw/Fail'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix (threshold={thresh:.2f})',
                fontsize=13, fontweight='bold')

    # ROC curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)
    ax.plot(fpr, tpr, linewidth=2, color='#e74c3c', label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)

    # Precision-Recall curve
    ax = axes[1, 0]
    prec_arr, rec_arr, thresholds = precision_recall_curve(yt, yp)
    ax.plot(rec_arr, prec_arr, linewidth=2, color='#2ecc71')
    ax.axvline(x=recall_score(yt, yhat), color='#e74c3c', linestyle='--',
               label=f'Operating point (t={thresh:.2f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend()

    # Threshold analysis
    ax = axes[1, 1]
    threshs = np.arange(0.1, 0.9, 0.01)
    recalls, precisions, f1s = [], [], []
    for t in threshs:
        yh = (yp >= t).astype(int)
        recalls.append(recall_score(yt, yh))
        precisions.append(precision_score(yt, yh, zero_division=0))
        f1s.append(f1_score(yt, yh, zero_division=0))

    ax.plot(threshs, recalls, linewidth=2, label='Recall', color='#e74c3c')
    ax.plot(threshs, precisions, linewidth=2, label='Precision',
            color='#3498db')
    ax.plot(threshs, f1s, linewidth=2, label='F1', color='#2ecc71')
    ax.axvline(x=thresh, color='black', linestyle='--',
               label=f'Selected threshold ({thresh:.2f})')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Threshold Analysis', fontsize=13, fontweight='bold')
    ax.legend()

    rec_final = recall_score(yt, yhat)
    prec_final = precision_score(yt, yhat)
    f1_final = f1_score(yt, yhat)

    fig.suptitle(
        f'Model Performance ({best}): '
        f'Recall={rec_final:.3f}, Precision={prec_final:.3f}, '
        f'F1={f1_final:.3f}, AUC={auc:.3f}',
        fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    fig.savefig(output_dir / 'model_performance.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved model_performance.png")

    return {
        'confusion_matrix': cm,
        'recall': rec_final, 'precision': prec_final,
        'f1': f1_final, 'auc': auc
    }


# ===================================================================
# 7. TEMPORAL STABILITY ANALYSIS
# ===================================================================

def temporal_stability_analysis(tables: dict, features_6: pd.DataFrame,
                                 output_dir: Path):
    """
    Compare model performance at week 3 vs week 6.
    Shows that predictions improve as more data comes in.
    """
    print("\n  Temporal stability: training at week 3 vs week 6...")

    # Week 3 features
    weekly_3 = compute_weekly_features(tables, max_week=EARLY_WEEK)
    features_3 = build_prediction_features(weekly_3, tables,
                                            cutoff_week=EARLY_WEEK)

    week_results = {}
    for week_label, feat_df in [('Week 3', features_3),
                                 ('Week 6', features_6)]:
        fc = get_feature_columns(feat_df)
        X = feat_df[fc].values
        y = feat_df['target'].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_yt, all_yp = [], []
        for tr, te in skf.split(X, y):
            m = RandomForestClassifier(n_estimators=200, max_depth=10,
                                        class_weight='balanced',
                                        random_state=42, n_jobs=-1)
            m.fit(X[tr], y[tr])
            all_yt.extend(y[te].tolist())
            all_yp.extend(m.predict_proba(X[te])[:, 1].tolist())

        yt = np.array(all_yt)
        yp = np.array(all_yp)
        auc = roc_auc_score(yt, yp)
        yhat = (yp >= 0.5).astype(int)
        week_results[week_label] = {
            'auc': auc,
            'recall': recall_score(yt, yhat),
            'precision': precision_score(yt, yhat),
            'f1': f1_score(yt, yhat),
            'y_true': yt, 'y_prob': yp
        }
        print(f"    {week_label}: AUC={auc:.4f}, "
              f"Recall={recall_score(yt, yhat):.4f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC comparison
    ax = axes[0]
    for label, res in week_results.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        ax.plot(fpr, tpr, linewidth=2,
                label=f'{label} (AUC={res["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC: Week 3 vs Week 6 Predictions', fontsize=13,
                 fontweight='bold')
    ax.legend()

    # Bar comparison
    ax = axes[1]
    metrics = ['auc', 'recall', 'precision', 'f1']
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2,
           [week_results['Week 3'][m] for m in metrics],
           w, label='Week 3', color='#f39c12')
    ax.bar(x + w/2,
           [week_results['Week 6'][m] for m in metrics],
           w, label='Week 6', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_title('Prediction Improvement: Week 3 vs Week 6',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / 'temporal_stability.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved temporal_stability.png")

    return week_results


# ===================================================================
# 8. FAIRNESS AUDIT
# ===================================================================

def fairness_audit(model_results: dict, output_dir: Path):
    """
    Break down prediction performance by demographic groups.
    Flags any significant disparities.
    """
    df = model_results['features_df'].copy()
    feat_cols = model_results['feature_cols']
    model = model_results['final_model']
    best_name = model_results['best_model_name']
    scaler = model_results['scaler']
    thresh = model_results['threshold']

    X = df[feat_cols].values
    if best_name == 'LogisticRegression':
        probs = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    df['pred_prob'] = probs
    df['pred'] = (probs >= thresh).astype(int)

    # Re-load original demographics for readable labels
    info = pd.read_csv(default_data_dir / 'studentInfo.csv')
    SK = ['code_module', 'code_presentation', 'id_student']
    demo_raw = info[SK + ['gender', 'age_band', 'imd_band']].drop_duplicates()
    df = df.merge(demo_raw, on=SK, how='left', suffixes=('_enc', ''))

    audit_groups = ['gender', 'age_band']
    available = [g for g in audit_groups if g in df.columns
                 and df[g].nunique() > 1]

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    audit_results = {}
    for i, group in enumerate(available):
        ax = axes[i]
        rows = []
        for val, sub in df.groupby(group):
            if len(sub) < 20:
                continue
            yt = sub['target'].values
            yhat = sub['pred'].values
            yp = sub['pred_prob'].values
            rows.append({
                'group': str(val),
                'n': len(sub),
                'base_rate': yt.mean(),
                'recall': recall_score(yt, yhat) if yt.sum() > 0 else 0,
                'precision': precision_score(yt, yhat)
                if yhat.sum() > 0 else 0,
                'auc': roc_auc_score(yt, yp)
                if len(np.unique(yt)) > 1 else 0
            })

        rdf = pd.DataFrame(rows)
        audit_results[group] = rdf

        x = np.arange(len(rdf))
        w = 0.25
        ax.bar(x - w, rdf['recall'], w, label='Recall', color='#e74c3c')
        ax.bar(x, rdf['precision'], w, label='Precision', color='#3498db')
        ax.bar(x + w, rdf['auc'], w, label='AUC', color='#2ecc71')
        ax.set_xticks(x)
        ax.set_xticklabels(rdf['group'], fontsize=9)
        ax.set_title(f'Fairness: {group.replace("_", " ").title()}',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)

        print(f"\n  Fairness by {group}:")
        for _, r in rdf.iterrows():
            print(f"    {r['group']:15s}: n={int(r['n']):5d}, "
                  f"base={r['base_rate']:.3f}, recall={r['recall']:.3f}, "
                  f"AUC={r['auc']:.3f}")

    plt.tight_layout()
    fig.savefig(output_dir / 'fairness_audit.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved fairness_audit.png")

    return audit_results


# ===================================================================
# 9. COST-BENEFIT ANALYSIS
# ===================================================================

def cost_benefit_analysis(model_results: dict, output_dir: Path,
                           cost_fn: float = 5000,
                           cost_fp: float = 50):
    """
    Frame threshold decision in cost terms.
    cost_fn: cost of missing an at-risk student (dropout cost)
    cost_fp: cost of unnecessary advisor outreach
    """
    best = model_results['best_model_name']
    yt = model_results['cv_results'][best]['y_true']
    yp = model_results['cv_results'][best]['y_prob']

    threshs = np.arange(0.10, 0.90, 0.01)
    costs = []
    for t in threshs:
        yhat = (yp >= t).astype(int)
        fn = ((yt == 1) & (yhat == 0)).sum()
        fp = ((yt == 0) & (yhat == 1)).sum()
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append({
            'threshold': t,
            'false_negatives': fn,
            'false_positives': fp,
            'total_cost': total_cost,
            'cost_missed_students': fn * cost_fn,
            'cost_unnecessary_outreach': fp * cost_fp
        })

    cost_df = pd.DataFrame(costs)
    opt_row = cost_df.loc[cost_df['total_cost'].idxmin()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cost_df['threshold'], cost_df['cost_missed_students'] / 1000,
            linewidth=2, label='Cost: Missed Students', color='#e74c3c')
    ax.plot(cost_df['threshold'], cost_df['cost_unnecessary_outreach'] / 1000,
            linewidth=2, label='Cost: Unnecessary Outreach', color='#3498db')
    ax.plot(cost_df['threshold'], cost_df['total_cost'] / 1000,
            linewidth=2.5, label='Total Cost', color='black', linestyle='--')
    ax.axvline(x=opt_row['threshold'], color='#2ecc71', linestyle=':',
               linewidth=2,
               label=f"Optimal threshold ({opt_row['threshold']:.2f})")

    ax.set_xlabel('Alert Threshold', fontsize=12)
    ax.set_ylabel('Cost (x $1,000)', fontsize=12)
    ax.set_title(
        f'Cost-Benefit Analysis\n'
        f'(Dropout cost=${cost_fn:,}, Outreach cost=${cost_fp:,}/student)',
        fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'cost_benefit.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Cost-optimal threshold: {opt_row['threshold']:.2f}")
    print(f"  Saved cost_benefit.png")

    return cost_df, opt_row


# ===================================================================
# 10. WHAT-IF COUNTERFACTUAL ANALYSIS
# ===================================================================

def whatif_analysis(model_results: dict, shap_results: dict,
                    output_dir: Path, n_students: int = 5):
    """
    For high-risk students, show: 'If feature X improved by Y%,
    risk would drop from A% to B%.'
    """
    df = model_results['features_df'].copy()
    feat_cols = model_results['feature_cols']
    model = model_results['final_model']
    best_name = model_results['best_model_name']
    scaler = model_results['scaler']
    top3 = shap_results['top3_features']

    X = df[feat_cols].values
    if best_name == 'LogisticRegression':
        probs = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    df['risk_prob'] = probs

    # Pick high-risk students
    high_risk = df[df['risk_prob'] > 0.6].sample(
        n=min(n_students, len(df[df['risk_prob'] > 0.6])),
        random_state=42)

    whatif_rows = []
    for _, student in high_risk.iterrows():
        sid = student['id_student']
        original_risk = student['risk_prob']

        for feat in top3:
            if feat not in feat_cols:
                continue

            fidx = feat_cols.index(feat)
            x_mod = df[feat_cols].loc[[student.name]].values.copy()

            # Improve the feature by 50%
            original_val = x_mod[0, fidx]
            # Direction depends on feature (higher clicks = better,
            # higher days_since_last = worse)
            if 'days_since_last' in feat or 'inactive' in feat:
                x_mod[0, fidx] = original_val * 0.5  # reduce by 50%
                change_desc = "reduced by 50%"
            else:
                x_mod[0, fidx] = original_val * 1.5  # increase by 50%
                change_desc = "increased by 50%"

            if best_name == 'LogisticRegression':
                new_risk = model.predict_proba(
                    scaler.transform(x_mod))[:, 1][0]
            else:
                new_risk = model.predict_proba(x_mod)[:, 1][0]

            whatif_rows.append({
                'student_id': int(sid),
                'feature': feat,
                'original_value': round(original_val, 2),
                'change': change_desc,
                'original_risk': round(original_risk * 100, 1),
                'new_risk': round(new_risk * 100, 1),
                'risk_reduction': round(
                    (original_risk - new_risk) * 100, 1)
            })

    whatif_df = pd.DataFrame(whatif_rows)

    print("\n  What-If Counterfactual Analysis (sample):")
    for _, r in whatif_df.head(9).iterrows():
        print(f"    Student {r['student_id']}: "
              f"If {r['feature']} {r['change']}, "
              f"risk drops {r['original_risk']}% -> {r['new_risk']}% "
              f"(reduction: {r['risk_reduction']}pp)")

    whatif_df.to_csv(output_dir / 'whatif_analysis.csv', index=False)
    print("  Saved whatif_analysis.csv")

    return whatif_df


# ===================================================================
# 11. STAFF NOTIFICATION DESIGN
# ===================================================================

def generate_staff_notification_spec(model_results: dict,
                                      shap_results: dict,
                                      output_dir: Path):
    """
    Generate the staff notification specification + sample alert table.
    """
    df = model_results['features_df'].copy()
    feat_cols = model_results['feature_cols']
    model = model_results['final_model']
    best_name = model_results['best_model_name']
    scaler = model_results['scaler']
    thresh = model_results['threshold']

    X = df[feat_cols].values
    if best_name == 'LogisticRegression':
        probs = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    df['risk_prob'] = probs
    df['risk_tier'] = pd.cut(
        df['risk_prob'],
        bins=[0, 0.5, 0.75, 1.0],
        labels=['Green', 'Amber', 'Red'])

    # Get top contributing feature per student from SHAP
    sv = shap_results['shap_values']
    top_driver_idx = np.argmax(np.abs(sv), axis=1)
    df['top_risk_driver'] = [feat_cols[i] for i in top_driver_idx]

    # Sample alert table: 10 students across tiers
    sample_alerts = []
    for tier in ['Red', 'Amber', 'Green']:
        tier_df = df[df['risk_tier'] == tier]
        n = min(4 if tier == 'Red' else 3, len(tier_df))
        if n > 0:
            sample_alerts.append(tier_df.sample(n, random_state=42))

    alert_df = pd.concat(sample_alerts).sort_values('risk_prob',
                                                      ascending=False)
    alert_table = alert_df[['id_student', 'code_module', 'risk_prob',
                            'risk_tier', 'top_risk_driver',
                            'final_result']].copy()
    alert_table['risk_prob'] = (alert_table['risk_prob'] * 100).round(1)
    alert_table.columns = ['Student ID', 'Course', 'Risk (%)', 'Tier',
                           'Primary Risk Factor', 'Actual Outcome']

    # Save notification spec
    spec = {
        'alert_system': 'PathAI Early Warning System',
        'alert_timing': 'Generated every Monday at 6:00 AM, '
                        'covering the prior 7 days of activity',
        'alert_tiers': {
            'Red': {
                'threshold': '>75% predicted risk',
                'action': 'Immediate outreach within 48 hours. '
                          'Schedule 1-on-1 meeting with student.',
                'notification': 'Email + in-app banner + SMS to advisor'
            },
            'Amber': {
                'threshold': '50-75% predicted risk',
                'action': 'Flag for weekly review. '
                          'Send supportive check-in email to student.',
                'notification': 'Email + in-app banner to advisor'
            },
            'Green': {
                'threshold': '<50% predicted risk',
                'action': 'No action required. '
                          'Available in dashboard for reference.',
                'notification': 'Dashboard only'
            }
        },
        'alert_content': [
            'Student name and ID',
            'Course and cohort',
            'Current risk score (0-100%)',
            'Risk tier (Red/Amber/Green)',
            'Trend arrow (improving/stable/declining)',
            'Top 3 contributing risk factors with plain-language explanation',
            'Recommended action based on tier',
            'Link to full student engagement profile'
        ],
        'threshold_rationale': f'Alert threshold set at {thresh:.0%} '
                               f'to optimize for high recall (catching '
                               f'at-risk students) while maintaining '
                               f'acceptable precision (minimizing '
                               f'unnecessary outreach).',
        'sample_alert_count': {
            'Red': int((df['risk_tier'] == 'Red').sum()),
            'Amber': int((df['risk_tier'] == 'Amber').sum()),
            'Green': int((df['risk_tier'] == 'Green').sum())
        }
    }

    with open(output_dir / 'notification_spec.json', 'w') as f:
        json.dump(spec, f, indent=2)

    alert_table.to_csv(output_dir / 'sample_alerts.csv', index=False)

    # Plot the sample alert table
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    colors_map = {'Red': '#ffcccc', 'Amber': '#fff3cc', 'Green': '#ccffcc'}
    cell_colors = [[colors_map.get(str(row['Tier']), 'white')]
                   * len(alert_table.columns)
                   for _, row in alert_table.iterrows()]

    table = ax.table(
        cellText=alert_table.values,
        colLabels=alert_table.columns,
        loc='center', cellLoc='center',
        cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title('Sample Staff Alert Dashboard', fontsize=14,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / 'sample_alert_table.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved notification_spec.json, sample_alerts.csv, "
          "sample_alert_table.png")

    return spec, alert_table


# ===================================================================
# 12. STREAMLIT DASHBOARD (standalone file generation)
# ===================================================================

def generate_streamlit_dashboard(output_dir: Path):
    """Generate a self-contained Streamlit dashboard script."""
    dashboard_code = '''"""
PathAI Staff Alert Dashboard (Streamlit)
========================================
Run: streamlit run task2_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="PathAI Early Warning", layout="wide")

st.title("PathAI Early Warning Dashboard")
st.markdown("Weekly disengagement risk alerts for university advisors")

# Load data
output_dir = Path(__file__).resolve().parent.parent / "output" / "task2_output"
try:
    scores = pd.read_csv(output_dir / "engagement_scores_week6.csv")
    alerts = pd.read_csv(output_dir / "sample_alerts.csv")
except FileNotFoundError:
    st.error("Run task2_model.py first to generate the required data.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
tier_filter = st.sidebar.multiselect(
    "Risk Tier", ["Red", "Amber", "Green"], default=["Red", "Amber"])
course_filter = st.sidebar.selectbox(
    "Course", ["All"] + sorted(scores["code_module"].unique().tolist())
    if "code_module" in scores.columns else ["All"])

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
if "risk_tier" in scores.columns:
    col1.metric("Total Students", len(scores))
    col2.metric("Red Alerts",
                len(scores[scores.risk_tier == "Red"]),
                delta=None)
    col3.metric("Amber Alerts",
                len(scores[scores.risk_tier == "Amber"]))
    col4.metric("Green (Safe)",
                len(scores[scores.risk_tier == "Green"]))

# Alert table
st.subheader("Student Risk Alerts")
if len(alerts) > 0:
    filtered = alerts[alerts["Tier"].isin(tier_filter)]
    st.dataframe(filtered, use_container_width=True)

# Student lookup
st.subheader("Individual Student Lookup")
if "id_student" in scores.columns:
    sid = st.selectbox("Select Student ID",
                       sorted(scores["id_student"].unique()))
    stu = scores[scores.id_student == sid]
    if len(stu) > 0:
        row = stu.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score",
                  f"{row.get('risk_prob', 0)*100:.1f}%")
        c2.metric("Risk Tier", row.get("risk_tier", "N/A"))
        c3.metric("Top Risk Factor",
                  row.get("top_risk_driver", "N/A"))
'''
    dash_path = output_dir / 'task2_dashboard.py'
    with open(dash_path, 'w') as f:
        f.write(dashboard_code)
    print(f"  Saved Streamlit dashboard: {dash_path}")


# ===================================================================
# 13. MAIN PIPELINE
# ===================================================================

def run_task2(data_dir: Path = None, output_dir: Path = None):
    """Run the full Task 2 pipeline."""
    if data_dir is None:
        data_dir = default_data_dir
    if output_dir is None:
        output_dir = default_output_dir
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK 2: PREDICTIVE DISENGAGEMENT MODEL")
    print("=" * 60)

    # 1. Load data
    print("\n[1/11] Loading OULAD data...")
    tables = load_oulad(data_dir)

    # 2. Compute weekly features up to week 6
    print(f"\n[2/11] Computing weekly features (weeks 1-{PREDICTION_WEEK})...")
    weekly_6 = compute_weekly_features(tables, max_week=PREDICTION_WEEK)
    print(f"  Weekly features shape: {weekly_6.shape}")

    # 3. Build prediction feature matrix
    print("\n[3/11] Building prediction feature matrix (hybrid)...")
    features_df = build_prediction_features(weekly_6, tables,
                                             cutoff_week=PREDICTION_WEEK)
    features_df.to_csv(output_dir / 'prediction_features.csv', index=False)
    feat_cols = get_feature_columns(features_df)
    print(f"  Feature matrix: {features_df.shape} "
          f"({len(feat_cols)} features)")

    # 4. Train and evaluate models
    print("\n[4/11] Training models (Leave-One-Course-Out CV)...")
    model_results = train_and_evaluate_models(features_df, output_dir)

    # 5. Confusion matrix and metrics
    print("\n[5/11] Plotting confusion matrix and metrics...")
    metrics = plot_confusion_and_metrics(model_results, output_dir)

    # 6. Calibration analysis
    print("\n[6/11] Calibration analysis...")
    cal_results = calibration_analysis(model_results, output_dir)

    # 7. SHAP explainability
    print("\n[7/11] SHAP feature importance...")
    shap_results = shap_analysis(model_results, output_dir)

    # 8. Temporal stability
    print("\n[8/11] Temporal stability (week 3 vs 6)...")
    temporal_results = temporal_stability_analysis(tables, features_df,
                                                    output_dir)

    # 9. Fairness audit
    print("\n[9/11] Fairness audit...")
    fairness_results = fairness_audit(model_results, output_dir)

    # 10. Cost-benefit analysis
    print("\n[10/11] Cost-benefit analysis...")
    cost_df, opt_row = cost_benefit_analysis(model_results, output_dir)

    # What-if analysis
    print("\n  What-if counterfactual analysis...")
    whatif_df = whatif_analysis(model_results, shap_results, output_dir)

    # 11. Staff notification design
    print("\n[11/11] Staff notification design...")
    spec, alert_table = generate_staff_notification_spec(
        model_results, shap_results, output_dir)

    # Save engagement scores with risk for dashboard
    df = model_results['features_df'].copy()
    feat_cols = model_results['feature_cols']
    model = model_results['final_model']
    best_name = model_results['best_model_name']
    scaler = model_results['scaler']
    X = df[feat_cols].values
    if best_name == 'LogisticRegression':
        probs = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]
    df['risk_prob'] = probs
    df['risk_tier'] = pd.cut(df['risk_prob'], bins=[0, 0.5, 0.75, 1.0],
                              labels=['Green', 'Amber', 'Red'])
    sv = shap_results['shap_values']
    top_idx = np.argmax(np.abs(sv), axis=1)
    df['top_risk_driver'] = [feat_cols[i] for i in top_idx]
    df.to_csv(output_dir / 'engagement_scores_week6.csv', index=False)

    # Generate Streamlit dashboard
    generate_streamlit_dashboard(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("TASK 2 COMPLETE")
    print("=" * 60)
    print(f"\n  Best model:       {model_results['best_model_name']}")
    print(f"  AUC:              {metrics['auc']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  F1:               {metrics['f1']:.4f}")
    print(f"  Brier score:      {cal_results['brier_score']:.4f}")
    print(f"  Threshold:        {model_results['threshold']:.3f}")
    print(f"  Top 3 features:   {shap_results['top3_features']}")
    print(f"\n  Outputs saved to: {output_dir}")

    return {
        'model_results': model_results,
        'metrics': metrics,
        'calibration': cal_results,
        'shap': shap_results,
        'temporal': temporal_results,
        'fairness': fairness_results,
        'cost_benefit': (cost_df, opt_row),
        'whatif': whatif_df,
        'notification': (spec, alert_table)
    }


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2: Predictive Disengagement Model")
    parser.add_argument('--data-dir', type=str, default=str(default_data_dir),
                        help='Path to OULAD CSV files')
    parser.add_argument('--output-dir', type=str,
                        default=str(default_output_dir),
                        help='Path to save outputs')
    args = parser.parse_args()

    run_task2(data_dir=Path(args.data_dir),
              output_dir=Path(args.output_dir))
