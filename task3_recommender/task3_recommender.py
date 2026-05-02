"""
Task 3: Course Recommendation Engine
=====================================
PathAI Engine - Studor DS Screening Project

Given a student's profile and behavioral history, recommend the 3 most
suitable courses from the OULAD catalog. Includes:

- Content-based filtering (course success profiles + cosine similarity)
- Collaborative filtering (user-user KNN + SVD matrix factorization)
- Hybrid recommender with adaptive blending weight
- Multi-layer cold-start fallback (demographic -> popularity -> diversity)
- Engagement-style-aware matching (ties to Task 1)
- Risk-aware recommendations (ties to Task 2)
- Plain-language recommendation explanations
- Course difficulty profiling
- Evaluation: Precision@K, Coverage, Novelty

Author: Satyadev
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
default_data_dir = project_root / 'data'
default_output_dir = project_root / 'output' / 'task3_output'

RESULT_MAP = {'Distinction': 4, 'Pass': 3, 'Fail': 2, 'Withdrawn': 1}
ASSESSMENT_ACTIVITIES = {'quiz', 'externalquiz', 'questionnaire'}
CONTENT_ACTIVITIES = {
    'oucontent', 'resource', 'url', 'homepage', 'subpage',
    'glossary', 'htmlactivity', 'page', 'folder'}

PROFILE_FEATURES = [
    'total_clicks', 'active_days', 'n_sites_visited', 'content_clicks',
    'assess_clicks', 'forum_clicks', 'activity_entropy',
    'content_ratio', 'forum_ratio', 'mean_score', 'n_submissions']


# ===================================================================
# 1. DATA LOADING & PROFILES
# ===================================================================

def load_oulad(data_dir):
    tables = {}
    for f in ['courses', 'vle', 'assessments', 'studentInfo',
              'studentVle', 'studentAssessment', 'studentRegistration']:
        tables[f] = pd.read_csv(Path(data_dir) / f"{f}.csv")
        print(f"  {f}: {tables[f].shape}")
    return tables


def build_student_profiles(tables):
    """Build behavioral + academic profile per student-course pair."""
    svle = tables['studentVle'].copy()
    vle = tables['vle'][['id_site', 'code_module', 'code_presentation',
                          'activity_type']].copy()
    info = tables['studentInfo'].copy()
    sa = tables['studentAssessment'].copy()
    assessments = tables['assessments'].copy()

    svle = svle.merge(vle, on=['id_site', 'code_module', 'code_presentation'],
                      how='left')
    SK = ['code_module', 'code_presentation', 'id_student']

    # Clickstream aggregates
    agg = svle.groupby(SK).agg(
        total_clicks=('sum_click', 'sum'),
        active_days=('date', 'nunique'),
        n_sites_visited=('id_site', 'nunique'),
    ).reset_index()

    # Activity type clicks
    svle['is_content'] = svle['activity_type'].isin(CONTENT_ACTIVITIES).astype(int)
    svle['is_assess'] = svle['activity_type'].isin(ASSESSMENT_ACTIVITIES).astype(int)
    svle['is_forum'] = (svle['activity_type'] == 'forumng').astype(int)

    content_c = svle[svle.is_content == 1].groupby(SK)['sum_click'].sum().reset_index().rename(columns={'sum_click': 'content_clicks'})
    assess_c = svle[svle.is_assess == 1].groupby(SK)['sum_click'].sum().reset_index().rename(columns={'sum_click': 'assess_clicks'})
    forum_c = svle[svle.is_forum == 1].groupby(SK)['sum_click'].sum().reset_index().rename(columns={'sum_click': 'forum_clicks'})

    # Entropy
    tc = svle.groupby(SK + ['activity_type'])['sum_click'].sum().reset_index()
    tt = tc.groupby(SK)['sum_click'].transform('sum')
    tc['prob'] = tc['sum_click'] / tt
    tc['neg_plogp'] = -tc['prob'] * np.log(tc['prob'] + 1e-10)
    entropy_agg = tc.groupby(SK)['neg_plogp'].sum().reset_index().rename(
        columns={'neg_plogp': 'activity_entropy'})

    # Scores
    sa_info = sa.merge(assessments[['id_assessment', 'code_module', 'code_presentation']],
                       on='id_assessment', how='left')
    score_agg = sa_info.groupby(SK).agg(
        mean_score=('score', 'mean'),
        n_submissions=('score', 'count'),
    ).reset_index()

    # Merge
    profiles = agg.copy()
    for df in [content_c, assess_c, forum_c, entropy_agg, score_agg]:
        profiles = profiles.merge(df, on=SK, how='left')

    demo_cols = ['gender', 'region', 'highest_education', 'imd_band',
                 'age_band', 'num_of_prev_attempts', 'studied_credits',
                 'disability', 'final_result']
    profiles = profiles.merge(info[SK + demo_cols].drop_duplicates(),
                               on=SK, how='left')

    # Derived
    total_typed = profiles['content_clicks'].fillna(0) + profiles['assess_clicks'].fillna(0)
    profiles['content_ratio'] = np.where(total_typed > 0,
        profiles['content_clicks'].fillna(0) / total_typed, 0.5)
    profiles['forum_ratio'] = np.where(profiles['total_clicks'].fillna(0) > 0,
        profiles['forum_clicks'].fillna(0) / profiles['total_clicks'].fillna(1), 0)
    profiles['result_score'] = profiles['final_result'].map(RESULT_MAP).fillna(2)
    profiles = profiles.fillna(0)

    print(f"  Student profiles: {profiles.shape}")
    return profiles


def build_course_profiles(profiles, tables):
    """Build profile per course module."""
    vle = tables['vle']; assessments = tables['assessments']
    rows = []
    for module in profiles['code_module'].unique():
        md = profiles[profiles['code_module'] == module]
        succ = md[md['final_result'].isin(['Pass', 'Distinction'])]
        n = len(md); n_pass = len(succ)
        pr = n_pass / max(n, 1)
        wr = len(md[md.final_result == 'Withdrawn']) / max(n, 1)

        src = succ if len(succ) > 0 else md
        mv = vle[vle['code_module'] == module]
        nr = len(mv)
        tc = mv['activity_type'].value_counts()

        ma = assessments[assessments['code_module'] == module]

        rows.append({
            'code_module': module,
            'pass_rate': pr, 'withdraw_rate': wr,
            'avg_clicks_successful': src['total_clicks'].mean(),
            'avg_active_days_successful': src['active_days'].mean(),
            'avg_score': src['mean_score'].mean(),
            'n_resources': nr,
            'content_heavy': tc.get('oucontent', 0) / max(nr, 1),
            'assess_heavy': (tc.get('quiz', 0) + tc.get('externalquiz', 0)) / max(nr, 1),
            'forum_heavy': tc.get('forumng', 0) / max(nr, 1),
            'n_assessments': ma['id_assessment'].nunique(),
            'has_exam': int((ma['assessment_type'] == 'Exam').any()),
            'success_content_ratio': src['content_ratio'].mean() if 'content_ratio' in src else 0.5,
            'success_entropy': src['activity_entropy'].mean() if 'activity_entropy' in src else 1.0,
            'success_forum_ratio': src['forum_ratio'].mean() if 'forum_ratio' in src else 0.1,
            'n_students': n,
            'difficulty_score': round((1 - pr) * 100, 1),
        })

    cp = pd.DataFrame(rows)
    print(f"  Course profiles: {cp.shape}")
    return cp


# ===================================================================
# 2. CONTENT-BASED FILTERING
# ===================================================================

def content_based_recommend(stu_profile, profiles, course_profiles,
                             taken, top_k=3):
    available = [c for c in course_profiles['code_module'].unique()
                 if c not in taken]
    if not available:
        return []

    stu_vec = stu_profile[PROFILE_FEATURES].values.astype(float).reshape(1, -1)
    recs = []
    for course in available:
        succ = profiles[(profiles['code_module'] == course) &
                        (profiles['final_result'].isin(['Pass', 'Distinction']))]
        if len(succ) < 3:
            succ = profiles[profiles['code_module'] == course]
        if len(succ) == 0:
            continue
        svec = succ[PROFILE_FEATURES].mean().values.reshape(1, -1)
        sim = cosine_similarity(stu_vec, svec)[0][0]
        cp = course_profiles[course_profiles['code_module'] == course].iloc[0]
        exp = (f"Your engagement style is {sim*100:.0f}% similar to successful "
               f"students in {course} (pass rate: {cp['pass_rate']*100:.0f}%)")
        recs.append((course, sim, exp))
    recs.sort(key=lambda x: -x[1])
    return recs[:top_k]


# ===================================================================
# 3. COLLABORATIVE FILTERING - KNN
# ===================================================================

def build_interaction_matrix(profiles):
    p = profiles.copy()
    sc = MinMaxScaler()
    rn = sc.fit_transform(p[['result_score']].values)
    cn = sc.fit_transform(p[['total_clicks']].values)
    p['composite_rating'] = 0.6 * rn.flatten() + 0.4 * cn.flatten()
    interaction = p.groupby(['id_student', 'code_module'])['composite_rating'].max().reset_index()
    matrix = interaction.pivot(index='id_student', columns='code_module',
                                values='composite_rating').fillna(0)
    return matrix, interaction


def knn_cf_recommend(sid, matrix, taken, k_neighbors=20, top_k=3):
    if sid not in matrix.index:
        return []
    available = [c for c in matrix.columns if c not in taken]
    if not available:
        return []

    sv = matrix.loc[sid].values.reshape(1, -1)
    k = min(k_neighbors, len(matrix) - 1)
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    knn.fit(matrix.values)
    dists, idxs = knn.kneighbors(sv)
    ni, nd = idxs[0][1:], dists[0][1:]
    w = np.maximum(1 - nd, 0)

    recs = []
    for course in available:
        nr = matrix.iloc[ni][course].values
        ws = np.dot(w, nr) / max(w.sum(), 1e-9)
        nt = (nr > 0).sum()
        exp = f"{nt} similar students took {course}, weighted score: {ws:.2f}/1.00"
        recs.append((course, ws, exp))
    recs.sort(key=lambda x: -x[1])
    return recs[:top_k]


# ===================================================================
# 4. COLLABORATIVE FILTERING - SVD
# ===================================================================

def svd_cf_recommend(sid, matrix, taken, n_components=5, top_k=3):
    if sid not in matrix.index:
        return []
    available = [c for c in matrix.columns if c not in taken]
    if not available:
        return []

    nc = min(n_components, min(matrix.shape) - 1)
    if nc < 1:
        return []

    svd = TruncatedSVD(n_components=nc, random_state=42)
    uf = svd.fit_transform(matrix.values)
    predicted = uf @ svd.components_

    sidx = list(matrix.index).index(sid)
    all_c = matrix.columns.tolist()

    recs = []
    for course in available:
        cidx = all_c.index(course)
        pr = np.clip(predicted[sidx, cidx], 0, 1)
        exp = f"SVD predicted affinity for {course}: {pr:.2f}/1.00"
        recs.append((course, pr, exp))
    recs.sort(key=lambda x: -x[1])
    return recs[:top_k]


# ===================================================================
# 5. COLD-START
# ===================================================================

def cold_start_recommend(demo, profiles, course_profiles, top_k=3):
    recs = []
    demo_feats = ['age_band', 'highest_education', 'studied_credits']
    avail = [f for f in demo_feats if f in demo and demo[f] is not None]

    if avail:
        mask = pd.Series(True, index=profiles.index)
        for f in avail:
            mask = mask & (profiles[f] == demo[f])
        similar = profiles[mask]
        if len(similar) < 10 and 'highest_education' in demo:
            similar = profiles[profiles['highest_education'] == demo['highest_education']]

        if len(similar) > 0:
            succ = similar[similar['final_result'].isin(['Pass', 'Distinction'])]
            if len(succ) > 0:
                cs = succ.groupby('code_module').agg(
                    n_success=('id_student', 'count'),
                    avg_score=('mean_score', 'mean')).reset_index()
                cs['score'] = cs['n_success'] / cs['n_success'].max() * 0.6 + cs['avg_score'] / 100 * 0.4
                cs = cs.sort_values('score', ascending=False)
                for _, r in cs.iterrows():
                    exp = (f"Students with your background succeed in {r['code_module']} "
                           f"({int(r['n_success'])} similar students passed)")
                    recs.append((r['code_module'], r['score'], exp))

    # Popularity fallback
    if len(recs) < top_k:
        already = {r[0] for r in recs}
        for _, r in course_profiles.sort_values('pass_rate', ascending=False).iterrows():
            if r['code_module'] not in already:
                exp = (f"{r['code_module']} has {r['pass_rate']*100:.0f}% pass rate "
                       f"({int(r['n_students'])} students)")
                recs.append((r['code_module'], r['pass_rate'], exp))
                already.add(r['code_module'])

    # Diversity injection
    if len(recs) >= top_k:
        top_c = {r[0] for r in recs[:top_k]}
        top_diff = course_profiles[course_profiles['code_module'].isin(top_c)]['difficulty_score'].mean()
        for _, r in course_profiles.iterrows():
            if r['code_module'] not in top_c and abs(r['difficulty_score'] - top_diff) > 10:
                exp = (f"Exploratory pick: {r['code_module']} offers a different "
                       f"challenge level (difficulty: {r['difficulty_score']:.0f}/100)")
                recs[top_k - 1] = (r['code_module'], recs[top_k - 1][1] * 0.9, exp)
                break

    return recs[:top_k]


# ===================================================================
# 6. HYBRID
# ===================================================================

def hybrid_recommend(sid, stu_profile, profiles, course_profiles,
                      matrix, taken, top_k=3):
    n_taken = len(taken)
    alpha = 0.3 if n_taken >= 3 else 0.5 if n_taken >= 2 else 0.7

    cb = content_based_recommend(stu_profile, profiles, course_profiles, taken, 10)
    knn = knn_cf_recommend(sid, matrix, taken, top_k=10)
    svd = svd_cf_recommend(sid, matrix, taken, top_k=10)

    # Normalize and blend
    cb_d = {c: s for c, s, e in cb}
    cb_e = {c: e for c, s, e in cb}
    cf_d = {}
    cf_e = {}
    for c, s, e in knn:
        cf_d[c] = s
        cf_e[c] = e
    for c, s, e in svd:
        cf_d[c] = (cf_d.get(c, 0) + s) / 2
        if c not in cf_e:
            cf_e[c] = e

    def _norm(d):
        if not d:
            return {}
        mx, mn = max(d.values()), min(d.values())
        r = mx - mn if mx > mn else 1
        return {k: (v - mn) / r for k, v in d.items()}

    cb_n = _norm(cb_d)
    cf_n = _norm(cf_d)

    all_c = set(cb_n) | set(cf_n)
    hybrid = []
    for c in all_c:
        hs = alpha * cb_n.get(c, 0) + (1 - alpha) * cf_n.get(c, 0)
        parts = []
        if c in cb_e:
            parts.append(f"Content: {cb_e[c]}")
        if c in cf_e:
            parts.append(f"Collab: {cf_e[c]}")
        hybrid.append((c, hs, " | ".join(parts)))

    hybrid.sort(key=lambda x: -x[1])
    return hybrid[:top_k]


# ===================================================================
# 7. ENRICHMENT (Style, Risk, Cards)
# ===================================================================

def engagement_style_scores(stu, course_profiles, taken):
    sc, sf, se = stu.get('content_ratio', 0.5), stu.get('forum_ratio', 0.1), stu.get('activity_entropy', 1.0)
    out = {}
    for _, c in course_profiles.iterrows():
        if c['code_module'] in taken:
            continue
        cm = 1 - abs(sc - c['success_content_ratio'])
        fm = 1 - abs(sf - c['success_forum_ratio'])
        em = max(0, min(1, 1 - abs((se - c['success_entropy']) / max(c['success_entropy'], 0.01))))
        out[c['code_module']] = {'score': 0.5 * cm + 0.3 * fm + 0.2 * em}
    return out


def risk_aware_scores(stu, profiles, taken):
    sv = stu[PROFILE_FEATURES].values.astype(float).reshape(1, -1)
    out = {}
    for course in profiles['code_module'].unique():
        if course in taken:
            continue
        cs = profiles[profiles['code_module'] == course]
        if len(cs) < 5:
            continue
        cvecs = cs[PROFILE_FEATURES].values.astype(float)
        sims = cosine_similarity(sv, cvecs)[0]
        topk = min(20, len(sims))
        ti = np.argsort(sims)[-topk:]
        sim_stu = cs.iloc[ti]
        nf = len(sim_stu[sim_stu['final_result'].isin(['Withdrawn', 'Fail'])])
        out[course] = {'risk': nf / len(sim_stu), 'safety_score': 1 - nf / len(sim_stu)}
    return out


def format_course_card(course, course_profiles, score, explanation,
                        style=None, risk=None):
    cp = course_profiles[course_profiles['code_module'] == course]
    if len(cp) == 0:
        return {'course': course, 'match_score': round(score * 100, 1), 'explanation': explanation}
    cp = cp.iloc[0]
    card = {
        'course': course,
        'match_score': round(score * 100, 1),
        'explanation': explanation,
        'pass_rate': round(cp['pass_rate'] * 100, 1),
        'difficulty': round(cp['difficulty_score'], 1),
        'avg_workload_clicks': round(cp['avg_clicks_successful'], 0),
        'n_assessments': int(cp['n_assessments']),
        'has_exam': bool(cp['has_exam']),
        'course_type': ('Content-heavy' if cp['content_heavy'] > 0.3
                        else 'Assessment-heavy' if cp['assess_heavy'] > 0.15
                        else 'Balanced'),
    }
    if style and course in style:
        card['style_match'] = round(style[course]['score'] * 100, 1)
    if risk and course in risk:
        card['predicted_risk'] = round(risk[course]['risk'] * 100, 1)
    return card


# ===================================================================
# 8. EVALUATION
# ===================================================================

def evaluate_recommenders(profiles, course_profiles, matrix, output_dir):
    sc = profiles.groupby('id_student')['code_module'].apply(set).reset_index()
    multi = sc[sc['code_module'].apply(len) >= 2]

    if len(multi) == 0:
        print("  No multi-course students. Using proxy evaluation.")
        return _proxy_eval(profiles, course_profiles, matrix, output_dir)

    print(f"  Evaluating on {len(multi)} multi-course students")
    all_c = set(profiles['code_module'].unique())
    res = {m: {'hits': 0, 'total': 0, 'recs': []}
           for m in ['content_based', 'knn_cf', 'svd_cf', 'hybrid']}

    for _, row in multi.iterrows():
        sid = row['id_student']
        clist = sorted(row['code_module'])
        held = clist[-1]
        train = set(clist[:-1])

        sd = profiles[(profiles['id_student'] == sid) & (profiles['code_module'].isin(train))]
        if len(sd) == 0:
            continue
        sp = sd.iloc[0]

        for method, func in [
            ('content_based', lambda: content_based_recommend(sp, profiles, course_profiles, train, 3)),
            ('knn_cf', lambda: knn_cf_recommend(sid, matrix, train, top_k=3)),
            ('svd_cf', lambda: svd_cf_recommend(sid, matrix, train, top_k=3)),
            ('hybrid', lambda: hybrid_recommend(sid, sp, profiles, course_profiles, matrix, train, 3)),
        ]:
            recs = func()
            rc = [r[0] for r in recs]
            res[method]['total'] += 1
            res[method]['recs'].extend(rc)
            if held in rc:
                res[method]['hits'] += 1

    metrics = {}
    for m, r in res.items():
        t = max(r['total'], 1)
        p3 = r['hits'] / t
        cov = len(set(r['recs'])) / len(all_c) if r['recs'] else 0
        metrics[m] = {'precision@3': round(p3, 4), 'coverage': round(cov, 4),
                      'n_evaluated': r['total']}
        print(f"    {m:20s}: P@3={p3:.4f}, Coverage={cov:.4f} ({r['total']} students)")

    _plot_eval(metrics, output_dir)
    return metrics


def _proxy_eval(profiles, course_profiles, matrix, output_dir):
    all_c = set(profiles['code_module'].unique())
    sample = profiles.drop_duplicates('id_student').sample(n=min(200, len(profiles)), random_state=42)
    hits, total, recs_list = 0, 0, []

    for _, stu in sample.iterrows():
        actual = stu['code_module']
        demo = {'age_band': stu.get('age_band'), 'highest_education': stu.get('highest_education'),
                'studied_credits': stu.get('studied_credits')}
        recs = cold_start_recommend(demo, profiles, course_profiles, top_k=3)
        rc = [r[0] for r in recs]
        total += 1; recs_list.extend(rc)
        if actual in rc:
            hits += 1

    p = hits / max(total, 1)
    cov = len(set(recs_list)) / len(all_c)
    metrics = {'cold_start': {'precision@3': round(p, 4), 'coverage': round(cov, 4), 'n_evaluated': total}}
    print(f"    Cold-start proxy: P@3={p:.4f}, Coverage={cov:.4f}")
    _plot_eval(metrics, output_dir)
    return metrics


def _plot_eval(metrics, output_dir):
    methods = list(metrics.keys())
    if not methods:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    ax = axes[0]
    vals = [metrics[m].get('precision@3', 0) for m in methods]
    bars = ax.bar(range(len(methods)), vals, color=colors[:len(methods)], width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Precision@3'); ax.set_title('Recommendation Accuracy', fontweight='bold')
    ax.set_ylim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.3f}', ha='center')

    ax = axes[1]
    vals = [metrics[m].get('coverage', 0) for m in methods]
    bars = ax.bar(range(len(methods)), vals, color=colors[:len(methods)], width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Catalog Coverage'); ax.set_title('Recommendation Diversity', fontweight='bold')
    ax.set_ylim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    fig.savefig(output_dir / 'evaluation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved evaluation_comparison.png")


def plot_course_profiles_viz(cp, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    cp_s = cp.sort_values('difficulty_score')
    cols = plt.cm.RdYlGn_r(cp_s['difficulty_score'] / 100)
    ax.barh(cp_s['code_module'], cp_s['difficulty_score'], color=cols)
    ax.set_xlabel('Difficulty Score'); ax.set_title('Course Difficulty', fontweight='bold')
    for i, (_, r) in enumerate(cp_s.iterrows()):
        ax.text(r['difficulty_score'] + 1, i, f"{r['pass_rate']*100:.0f}% pass", va='center', fontsize=9)

    ax = axes[1]
    x = np.arange(len(cp_s)); w = 0.25
    ax.bar(x - w, cp_s['content_heavy'], w, label='Content', color='#3498db')
    ax.bar(x, cp_s['assess_heavy'], w, label='Assessment', color='#e74c3c')
    ax.bar(x + w, cp_s['forum_heavy'], w, label='Forum', color='#2ecc71')
    ax.set_xticks(x); ax.set_xticklabels(cp_s['code_module'])
    ax.set_ylabel('Resource Proportion'); ax.set_title('Course Structure', fontweight='bold')
    ax.legend()

    ax = axes[2]
    ax.bar(cp_s['code_module'], cp_s['avg_clicks_successful'], color='#f39c12', width=0.6)
    ax.set_ylabel('Avg Clicks (Successful)'); ax.set_title('Course Workload', fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'course_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved course_profiles.png")


def plot_sample_recs(cards, output_dir):
    if not cards:
        return
    fig, ax = plt.subplots(figsize=(16, len(cards) * 0.6 + 2))
    ax.axis('off')
    headers = ['Student', 'Course', 'Match %', 'Style %', 'Risk %', 'Difficulty', 'Pass %', 'Type']
    rows, cc = [], []
    for c in cards:
        rows.append([
            c.get('student_id', ''), c.get('course', ''),
            f"{c.get('match_score', 0):.0f}", f"{c.get('style_match', 0):.0f}",
            f"{c.get('predicted_risk', 0):.0f}", f"{c.get('difficulty', 0):.0f}",
            f"{c.get('pass_rate', 0):.0f}", c.get('course_type', '')])
        risk = c.get('predicted_risk', 50)
        cc.append(['#ccffcc' if risk < 30 else '#fff3cc' if risk < 60 else '#ffcccc'] * len(headers))

    t = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center', cellColours=cc)
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    ax.set_title('Sample Course Recommendations', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(output_dir / 'sample_recommendations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sample_recommendations.png")


# ===================================================================
# 9. MAIN PIPELINE
# ===================================================================

def run_task3(data_dir=None, output_dir=None):
    if data_dir is None: data_dir = default_data_dir
    if output_dir is None: output_dir = default_output_dir
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK 3: COURSE RECOMMENDATION ENGINE")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    tables = load_oulad(data_dir)

    print("\n[2/7] Building student profiles...")
    profiles = build_student_profiles(tables)
    profiles.to_csv(output_dir / 'student_profiles.csv', index=False)

    print("\n[3/7] Building course profiles...")
    cp = build_course_profiles(profiles, tables)
    cp.to_csv(output_dir / 'course_profiles.csv', index=False)
    plot_course_profiles_viz(cp, output_dir)

    print("\n[4/7] Building interaction matrix...")
    matrix, _ = build_interaction_matrix(profiles)
    print(f"  Matrix: {matrix.shape}, Sparsity: {(matrix == 0).sum().sum() / matrix.size:.1%}")

    print("\n[5/7] Evaluating recommenders...")
    eval_metrics = evaluate_recommenders(profiles, cp, matrix, output_dir)

    print("\n[6/7] Generating sample recommendations...")
    sample = profiles.drop_duplicates('id_student').sample(n=min(5, len(profiles)), random_state=42)
    all_cards = []
    for _, stu in sample.iterrows():
        sid = stu['id_student']
        taken = set(profiles[profiles['id_student'] == sid]['code_module'].unique())
        recs = hybrid_recommend(sid, stu, profiles, cp, matrix, taken, 3)
        style = engagement_style_scores(stu, cp, taken)
        risk = risk_aware_scores(stu, profiles, taken)
        for course, score, exp in recs:
            card = format_course_card(course, cp, score, exp, style, risk)
            card['student_id'] = int(sid)
            all_cards.append(card)

    pd.DataFrame(all_cards).to_csv(output_dir / 'sample_recommendations.csv', index=False)
    plot_sample_recs(all_cards[:9], output_dir)

    print("\n[7/7] Cold-start demo...")
    cold = cold_start_recommend(
        {'age_band': '0-35', 'highest_education': 'A Level or Equivalent', 'studied_credits': 60},
        profiles, cp, 3)
    for c, s, e in cold:
        print(f"    {c}: {e}")

    summary = {
        'primary_user': 'Students seeking next-semester course recommendations',
        'approaches': ['Content-based (cosine similarity)', 'KNN collaborative filtering',
                       'SVD collaborative filtering', 'Hybrid (adaptive alpha)'],
        'cold_start': 'Multi-layer: demographic -> popularity -> diversity',
        'similarity_metric': 'Cosine similarity',
        'evaluation': eval_metrics,
        'differentiators': ['Engagement-style matching (Task 1)',
                           'Risk-aware recs (Task 2)',
                           'Plain-language explanations',
                           'Course difficulty profiling'],
    }
    with open(output_dir / 'task3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("TASK 3 COMPLETE")
    print("=" * 60)
    return {'profiles': profiles, 'course_profiles': cp, 'eval_metrics': eval_metrics,
            'sample_cards': all_cards, 'summary': summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: Course Recommendation Engine")
    parser.add_argument('--data-dir', type=str, default=str(default_data_dir))
    parser.add_argument('--output-dir', type=str, default=str(default_output_dir))
    args = parser.parse_args()
    run_task3(Path(args.data_dir), Path(args.output_dir))
