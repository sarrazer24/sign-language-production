
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# Connexions corps MediaPipe (33 kp, indices 0-32)
BODY_CONNECTIONS_MEDIAPIPE = [
    # Visage simplifié
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Épaules → coudes → poignets
    (11, 13), (13, 15),   # bras gauche
    (12, 14), (14, 16),   # bras droit
    # Torse
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Jambes
    (23, 25), (25, 27), (27, 29), (29, 31),  # gauche
    (24, 26), (26, 28), (28, 30), (30, 32),  # droite
]

# Connexions main (21 kp, même structure gauche et droite)
HAND_CONNECTIONS = [
    # Pouce
    (0,1),(1,2),(2,3),(3,4),
    # Index
    (0,5),(5,6),(6,7),(7,8),
    # Majeur
    (0,9),(9,10),(10,11),(11,12),
    # Annulaire
    (0,13),(13,14),(14,15),(15,16),
    # Auriculaire
    (0,17),(17,18),(18,19),(19,20),
    # Paume
    (5,9),(9,13),(13,17),
]

# ── Format OpenPose 25 kp (si votre dataset utilise ce format) ──
BODY_CONNECTIONS_OPENPOSE25 = [
    # Tête
    (0,1),(0,15),(0,16),(15,17),(16,18),
    # Cou → corps
    (1,2),(1,5),(1,8),
    # Bras droit
    (2,3),(3,4),
    # Bras gauche
    (5,6),(6,7),
    # Hanches
    (8,9),(8,12),
    # Jambe droite
    (9,10),(10,11),(11,22),(11,23),(11,24),
    # Jambe gauche
    (12,13),(13,14),(14,19),(14,20),(14,21),
]

def get_connections(n_body_kp):
    """Choisit automatiquement les connexions selon le nombre de kp corps."""
    if n_body_kp >= 33:
        return BODY_CONNECTIONS_MEDIAPIPE
    else:
        return BODY_CONNECTIONS_OPENPOSE25


# ══════════════════════════════════════════════════════════════════


def parse_sequence(seq_flat, pose_dim=453, kp_dim=3):
    """
    Entrée : seq_flat (T, pose_dim) — ex: (T, 453) pour 151 kp × 3
    Sortie : dict avec corps, main_g, main_d en (T, N_kp, 2)
    """
    T = seq_flat.shape[0]
    n_kp = pose_dim // kp_dim

    # Reshape en (T, N_kp, kp_dim)
    seq = seq_flat.reshape(T, n_kp, kp_dim)

    # Garder seulement x,y
    seq_2d = seq[:, :, :2]

    # Déduire la structure selon n_kp
    if n_kp == 543:
        # MediaPipe complet : 33 corps + 21 main G + 21 main D + 468 visage
        body   = seq_2d[:, 0:33]
        hand_l = seq_2d[:, 33:54]
        hand_r = seq_2d[:, 54:75]
    elif n_kp == 137:
        # How2Sign custom : 33 corps + 21 main G + 21 main D + 62 face
        body   = seq_2d[:, 0:33]
        hand_l = seq_2d[:, 33:54]
        hand_r = seq_2d[:, 54:75]
    elif n_kp == 151:
        # 25 corps + 21 main G + 21 main D + 84 face (OpenPose-like)
        body   = seq_2d[:, 0:25]
        hand_l = seq_2d[:, 25:46]
        hand_r = seq_2d[:, 46:67]
    elif n_kp == 75:
        # Corps + 2 mains seulement
        body   = seq_2d[:, 0:33]
        hand_l = seq_2d[:, 33:54]
        hand_r = seq_2d[:, 54:75]
    else:
        # Fallback : prendre les 25 premiers comme corps
        body   = seq_2d[:, :min(25, n_kp)]
        hand_l = seq_2d[:, min(25,n_kp):min(46,n_kp)] if n_kp > 25 else np.zeros((T,21,2))
        hand_r = seq_2d[:, min(46,n_kp):min(67,n_kp)] if n_kp > 46 else np.zeros((T,21,2))

    return {'body': body, 'hand_l': hand_l, 'hand_r': hand_r, 'n_kp': n_kp}


def normalize_part(seq_part, ref_seq=None):
    """
    seq_part : (T, N, 2)
    Normalise par rapport au centre et à l'échelle globale de la séquence.
    """
    ref = ref_seq if ref_seq is not None else seq_part

    # Masquer les points nuls (non détectés)
    mask = ~np.all(ref == 0, axis=-1)   # (T, N) bool

    all_pts = ref[mask]  # (M, 2)
    if len(all_pts) < 2:
        return seq_part

    center = np.median(all_pts, axis=0)
    scale  = np.percentile(np.abs(all_pts - center), 90)
    scale  = max(scale, 1e-6)

    out = (seq_part - center) / scale
    # Remettre les points nuls à NaN pour ne pas les dessiner
    zero_mask = np.all(seq_part == 0, axis=-1)
    out[zero_mask] = np.nan
    return out


def normalize_sequence_smart(parsed, ref_parsed=None):
    """
    Normalise corps + mains de façon cohérente.
    Si ref_parsed fourni, utilise ses stats (pour aligner prédit sur réel).
    """
    ref = ref_parsed if ref_parsed is not None else parsed

    # Normaliser le corps en priorité (définit l'échelle globale)
    body_norm  = normalize_part(parsed['body'],   ref['body'])
    handl_norm = normalize_part(parsed['hand_l'], ref['hand_l'])
    handr_norm = normalize_part(parsed['hand_r'], ref['hand_r'])

    return {'body': body_norm, 'hand_l': handl_norm, 'hand_r': handr_norm}




COLORS = {
    'orig_body':   '#00c9a7',   # vert menthe
    'orig_handl':  '#48cae4',   # bleu clair
    'orig_handr':  '#48cae4',
    'pred_body':   '#ff6b6b',   # rouge corail
    'pred_handl':  '#ffa552',   # orange
    'pred_handr':  '#ffa552',
    'error':       '#ffd166',   # jaune
    'bg':          '#0f1117',
    'bg_panel':    '#161b22',
    'grid':        '#21262d',
    'text':        '#c9d1d9',
    'muted':       '#484f58',
}

def draw_part(ax, kp_2d, connections, color, lw=1.8, dot_size=12, alpha=1.0):
    """
    kp_2d       : (N, 2) pour une frame
    connections : liste de (i, j)
    Dessine les os et les joints sur ax.
    """
    N = len(kp_2d)

    # Os
    for (i, j) in connections:
        if i >= N or j >= N:
            continue
        p1, p2 = kp_2d[i], kp_2d[j]
        if np.any(np.isnan([p1, p2])):
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, lw=lw, solid_capstyle='round',
                solid_joinstyle='round', alpha=alpha, zorder=2)

    # Joints
    for i, p in enumerate(kp_2d):
        if np.any(np.isnan(p)):
            continue
        s = dot_size * 1.5 if i in [0, 7, 4, 8] else dot_size
        ax.scatter(p[0], p[1], s=s, c=color, zorder=3,
                   edgecolors='white', linewidths=0.3, alpha=alpha)


def draw_skeleton_frame(ax, norm, is_pred, alpha=1.0, lw_body=2.0, lw_hand=1.2):
    """Dessine corps + mains pour une frame normalisée."""
    body   = norm['body']
    hand_l = norm['hand_l']
    hand_r = norm['hand_r']

    n_body = body.shape[0]
    conns  = get_connections(n_body)

    if is_pred:
        c_body, c_hand = COLORS['pred_body'], COLORS['pred_handr']
    else:
        c_body, c_hand = COLORS['orig_body'], COLORS['orig_handl']

    draw_part(ax, body,   conns,           c_body, lw=lw_body, dot_size=15, alpha=alpha)
    draw_part(ax, hand_l, HAND_CONNECTIONS, c_hand, lw=lw_hand, dot_size=8,  alpha=alpha)
    draw_part(ax, hand_r, HAND_CONNECTIONS, c_hand, lw=lw_hand, dot_size=8,  alpha=alpha)


def get_axis_limits(norm_orig, norm_pred):
    """Calcule des limites d'axes communes et propres."""
    all_pts = []
    for d in [norm_orig, norm_pred]:
        for key in ['body', 'hand_l', 'hand_r']:
            pts = d[key].reshape(-1, 2)
            valid = pts[~np.any(np.isnan(pts), axis=1)]
            if len(valid):
                all_pts.append(valid)

    if not all_pts:
        return -2, 2, -2, 2

    all_pts = np.vstack(all_pts)
    p1, p99 = np.percentile(all_pts, 1, axis=0), np.percentile(all_pts, 99, axis=0)
    margin  = (p99 - p1).max() * 0.25 + 0.3
    cx, cy  = (p1 + p99) / 2
    half    = (p99 - p1).max() / 2 + margin
    return cx - half, cx + half, cy - half, cy + half


def style_ax(ax, title='', xmin=-2, xmax=2, ymin=-2, ymax=2, invert_y=True):
    ax.set_facecolor(COLORS['bg_panel'])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(COLORS['grid'])
        sp.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=8, color=COLORS['muted'],
                     pad=4, fontfamily='monospace')



def visualize(pred_flat, target_flat,
              text='', mpjpe=None, dtw=None,
              pose_dim=453, kp_dim=3,
              n_frames=6,
              save_path=None, show=True):
    """
    Visualisation principale — 3 rangées × n_frames colonnes.

    Paramètres :
        pred_flat   : (T, pose_dim) numpy array — sorties du modèle
        target_flat : (T, pose_dim) numpy array — ground truth
        pose_dim    : 453 pour 151 kp×3, ajustez selon votre dataset
        n_frames    : nombre de frames à afficher (6 recommandé)
    """
    T = pred_flat.shape[0]

    # ── Parse ──
    parsed_orig = parse_sequence(target_flat, pose_dim, kp_dim)
    parsed_pred = parse_sequence(pred_flat,   pose_dim, kp_dim)

    # ── Normalise (prédit calé sur le réel) ──
    norm_orig = normalize_sequence_smart(parsed_orig)
    norm_pred = normalize_sequence_smart(parsed_pred, ref_parsed=parsed_orig)

    # ── Sélection des frames ──
    if T <= n_frames:
        fids = list(range(T))
    else:
        fids = [int(round(i * (T - 1) / (n_frames - 1))) for i in range(n_frames)]
    n_cols = len(fids)

    # ── Limites d'axes ──
    # Construire des normes par frame pour les limites
    xmin_g, xmax_g, ymin_g, ymax_g = get_axis_limits(
        {k: norm_orig[k][fids] for k in norm_orig},
        {k: norm_pred[k][fids] for k in norm_pred}
    )
    lim = dict(xmin=xmin_g, xmax=xmax_g, ymin=ymin_g, ymax=ymax_g)

    # ── Figure ──
    fig_w = max(14, n_cols * 2.6 + 1.2)
    fig_h = 9.5
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=COLORS['bg'])

    # Titre
    title_lines = []
    if text:
        title_lines.append(f'"{text}"')
    metrics = []
    if mpjpe is not None: metrics.append(f'MPJPE = {mpjpe:.4f}')
    if dtw   is not None: metrics.append(f'DTW = {dtw:.4f}')
    if metrics: title_lines.append('   |   '.join(metrics))
    if title_lines:
        fig.suptitle('\n'.join(title_lines),
                     fontsize=10, color=COLORS['text'],
                     fontfamily='monospace', y=0.99)

    # GridSpec : colonne 0 = labels, colonnes 1..n_cols = frames
    gs = GridSpec(3, n_cols + 1, figure=fig,
                  width_ratios=[0.06] + [1] * n_cols,
                  hspace=0.05, wspace=0.03,
                  left=0.03, right=0.99,
                  top=0.92, bottom=0.03)

    row_meta = [
        ('RÉEL',       False, COLORS['orig_body']),
        ('PRÉDIT',     True,  COLORS['pred_body']),
        ('SUPERPOSÉ',  None,  COLORS['error']),
    ]

    for row, (label, is_pred, label_col) in enumerate(row_meta):

        # ── Label de rangée ──
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.set_facecolor(COLORS['bg_panel'])
        ax_lbl.text(0.5, 0.5, label,
                    ha='center', va='center',
                    fontsize=7, fontweight='bold', color=label_col,
                    transform=ax_lbl.transAxes, rotation=90,
                    fontfamily='monospace')
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for sp in ax_lbl.spines.values():
            sp.set_edgecolor(COLORS['grid']); sp.set_linewidth(0.4)

        for ci, fi in enumerate(fids):
            ax = fig.add_subplot(gs[row, ci + 1])
            frame_title = f'f{fi}' if row == 0 else ''
            style_ax(ax, title=frame_title, **lim)

            # Récupérer la frame normalisée
            o = {k: norm_orig[k][fi] for k in norm_orig}
            p = {k: norm_pred[k][fi] for k in norm_pred}

            if is_pred is False:
                # Rangée 0 : réel uniquement
                draw_skeleton_frame(ax, o, is_pred=False)

            elif is_pred is True:
                # Rangée 1 : prédit uniquement
                draw_skeleton_frame(ax, p, is_pred=True)

            else:
                # Rangée 2 : superposé + lignes d'erreur
                draw_skeleton_frame(ax, o, is_pred=False, alpha=0.6, lw_body=1.5)
                draw_skeleton_frame(ax, p, is_pred=True,  alpha=0.6, lw_body=1.5)

                # Lignes d'erreur sur les joints clés du corps
                nb = min(len(o['body']), len(p['body']))
                for ki in range(nb):
                    po, pp = o['body'][ki], p['body'][ki]
                    if np.any(np.isnan([po, pp])): continue
                    err = np.linalg.norm(po - pp)
                    if err < 0.05: continue
                    alpha_err = min(err / 0.5, 1.0)
                    ax.plot([po[0], pp[0]], [po[1], pp[1]],
                            color=COLORS['error'],
                            lw=max(0.5, err * 2.5),
                            alpha=alpha_err * 0.8,
                            linestyle='--', zorder=1)

    # ── Légende ──
    handles = [
        mpatches.Patch(color=COLORS['orig_body'], label='Réel (GT)'),
        mpatches.Patch(color=COLORS['pred_body'], label='Prédit'),
        mpatches.Patch(color=COLORS['error'],     label='Erreur'),
    ]
    fig.legend(handles=handles, loc='lower right',
               fontsize=8, facecolor=COLORS['bg_panel'],
               edgecolor=COLORS['grid'], labelcolor=COLORS['text'],
               framealpha=0.9, ncol=3,
               bbox_to_anchor=(0.99, 0.005))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['bg'])
        print(f'Sauvegardé : {save_path}')
    if show:
        plt.show()
    plt.close(fig)


def visualize_animated(pred_flat, target_flat,
                        text='', pose_dim=453, kp_dim=3,
                        fps=8, save_gif=None, show=True):
    """
    Animation frame par frame — côte à côte Réel | Prédit | Superposé.
    """
    from matplotlib.animation import FuncAnimation

    T = pred_flat.shape[0]
    parsed_orig = parse_sequence(target_flat, pose_dim, kp_dim)
    parsed_pred = parse_sequence(pred_flat,   pose_dim, kp_dim)
    norm_orig   = normalize_sequence_smart(parsed_orig)
    norm_pred   = normalize_sequence_smart(parsed_pred, ref_parsed=parsed_orig)

    xmin, xmax, ymin, ymax = get_axis_limits(norm_orig, norm_pred)
    lim = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=COLORS['bg'])
    titles = ['RÉEL (GT)', 'PRÉDIT', 'SUPERPOSÉ']
    cols   = [COLORS['orig_body'], COLORS['pred_body'], COLORS['error']]

    for ax, t, c in zip(axes, titles, cols):
        style_ax(ax, **lim)
        ax.set_title(t, color=c, fontsize=9, fontfamily='monospace', pad=5)

    frame_txt = fig.text(0.5, 0.01, '', ha='center',
                          color=COLORS['muted'], fontsize=9, fontfamily='monospace')
    if text:
        fig.suptitle(f'"{text}"', color=COLORS['text'],
                     fontsize=9, fontfamily='monospace')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    def update(fi):
        for ax, t, c in zip(axes, titles, cols):
            ax.cla()
            style_ax(ax, **lim)
            ax.set_title(t, color=c, fontsize=9, fontfamily='monospace', pad=5)

        o = {k: norm_orig[k][fi] for k in norm_orig}
        p = {k: norm_pred[k][fi] for k in norm_pred}

        draw_skeleton_frame(axes[0], o, is_pred=False)
        draw_skeleton_frame(axes[1], p, is_pred=True)
        draw_skeleton_frame(axes[2], o, is_pred=False, alpha=0.55, lw_body=1.5)
        draw_skeleton_frame(axes[2], p, is_pred=True,  alpha=0.55, lw_body=1.5)

        frame_txt.set_text(f'frame {fi:03d} / {T-1:03d}')
        return []

    anim = FuncAnimation(fig, update, frames=T,
                          interval=1000 // fps, blit=False)
    if save_gif:
        anim.save(save_gif, writer='pillow', fps=fps,
                  savefig_kwargs={'facecolor': COLORS['bg']})
        print(f'GIF sauvegardé : {save_gif}')
    if show:
        plt.show()
    plt.close(fig)
    return anim




def visualize_error_heatmap(pred_flat, target_flat,
                             pose_dim=453, kp_dim=3,
                             save_path=None, show=True):
    """Heatmap de l'erreur L2 par joint et par frame."""
    T  = pred_flat.shape[0]
    nk = pose_dim // kp_dim
    pred_2d   = pred_flat.reshape(T, nk, kp_dim)[:, :, :2]
    target_2d = target_flat.reshape(T, nk, kp_dim)[:, :, :2]

    # Garder les 75 premiers kp (corps + mains)
    N = min(75, nk)
    err = np.linalg.norm(pred_2d[:, :N] - target_2d[:, :N], axis=-1)  # (T, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={'width_ratios': [3, 1]},
                                    facecolor=COLORS['bg'])

    # Heatmap
    ax1.set_facecolor(COLORS['bg_panel'])
    im = ax1.imshow(err.T, aspect='auto', cmap='RdYlGn_r',
                    vmin=0, vmax=np.percentile(err, 95), interpolation='nearest')
    ax1.set_xlabel('Frame', color=COLORS['muted'], fontsize=9)
    ax1.set_ylabel('Keypoint index', color=COLORS['muted'], fontsize=9)
    ax1.set_title('Erreur L2 par joint et par frame',
                  color=COLORS['text'], fontsize=10, fontfamily='monospace')
    ax1.tick_params(colors=COLORS['muted'], labelsize=7)
    for sp in ax1.spines.values(): sp.set_edgecolor(COLORS['grid'])
    cbar = plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(colors=COLORS['muted'], labelsize=7)
    cbar.set_label('Erreur L2', color=COLORS['muted'], fontsize=8)

    # Erreur moyenne par joint
    ax2.set_facecolor(COLORS['bg_panel'])
    mean_err = err.mean(axis=0)
    norm_e   = mean_err / (mean_err.max() + 1e-8)
    colors_b = plt.cm.RdYlGn_r(norm_e)
    ax2.barh(range(N), mean_err, color=colors_b, height=0.7)
    ax2.set_title('Moy. par joint', color=COLORS['text'],
                  fontsize=9, fontfamily='monospace')
    ax2.set_xlabel('Erreur L2', color=COLORS['muted'], fontsize=8)
    ax2.tick_params(colors=COLORS['muted'], labelsize=7)
    ax2.set_ylim(-0.5, N - 0.5)
    ax2.invert_yaxis()
    for sp in ax2.spines.values(): sp.set_edgecolor(COLORS['grid'])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['bg'])
        print(f'Heatmap sauvegardée : {save_path}')
    if show:
        plt.show()
    plt.close(fig)




def visualize_mpjpe_curve(pred_flat, target_flat,
                           pose_dim=453, kp_dim=3,
                           text='', save_path=None, show=True):
    """Courbe MPJPE frame par frame."""
    T  = pred_flat.shape[0]
    nk = pose_dim // kp_dim
    p  = pred_flat.reshape(T, nk, kp_dim)[:, :, :2]
    t  = target_flat.reshape(T, nk, kp_dim)[:, :, :2]

    err_per_frame = np.linalg.norm(p - t, axis=-1).mean(axis=-1)  # (T,)
    avg = err_per_frame.mean()

    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg_panel'])

    ax.plot(err_per_frame, color=COLORS['orig_body'], lw=1.5, label='MPJPE / frame')
    ax.fill_between(range(T), err_per_frame, alpha=0.12, color=COLORS['orig_body'])
    ax.axhline(avg, color=COLORS['error'], lw=1.2,
               linestyle='--', label=f'Moy = {avg:.4f}')

    # Zone de best frames
    threshold = np.percentile(err_per_frame, 25)
    ax.fill_between(range(T),
                    np.where(err_per_frame <= threshold, err_per_frame, 0),
                    alpha=0.25, color='#00c9a7', label='Top 25% frames')

    ax.set_xlabel('Frame', color=COLORS['muted'], fontsize=9)
    ax.set_ylabel('MPJPE', color=COLORS['muted'], fontsize=9)
    ax.set_title(f'"{text}"   |   MPJPE = {avg:.4f}' if text else f'MPJPE = {avg:.4f}',
                 color=COLORS['text'], fontsize=9, fontfamily='monospace')
    ax.tick_params(colors=COLORS['muted'], labelsize=8)
    ax.legend(fontsize=8, facecolor=COLORS['bg_panel'],
              edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    for sp in ax.spines.values(): sp.set_edgecolor(COLORS['grid'])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['bg'])
        print(f'Courbe sauvegardée : {save_path}')
    if show:
        plt.show()
    plt.close(fig)




if __name__ == '__main__':
    print("Test autonome — données synthétiques réalistes...")

    rng  = np.random.default_rng(0)
    T    = 65
    POSE_DIM = 453   # 151 kp × 3 — comme votre dataset
    NKP  = POSE_DIM // 3  # 151

    # ── Construire une pose corps debout réaliste ──
    def make_standing_pose(NKP=151):
        """Pose de référence debout (OpenPose-like, kp 0-24)."""
        kp = np.zeros((NKP, 3))
       
        kp[0]  = [0.0,  0.0,  0.0] 
        kp[1]  = [0.0,  0.15, 0.0]  
        kp[2]  = [ 0.2, 0.15, 0.0]  
        kp[5]  = [-0.2, 0.15, 0.0]  
        kp[3]  = [ 0.35, 0.5, 0.0]   
        kp[6]  = [-0.35, 0.5, 0.0]  
        kp[4]  = [ 0.45, 0.85, 0.0] 
        kp[7]  = [-0.45, 0.85, 0.0]  
        kp[8]  = [ 0.0,  0.7, 0.0]  
        kp[9]  = [ 0.15, 0.7, 0.0] 
        kp[12] = [-0.15, 0.7, 0.0]   
        kp[10] = [ 0.15, 1.2, 0.0]  
        kp[13] = [-0.15, 1.2, 0.0]  
        kp[11] = [ 0.15, 1.7, 0.0]  
        kp[14] = [-0.15, 1.7, 0.0]  
        kp[15] = [ 0.05,-0.12,0.0]   
        kp[16] = [-0.05,-0.12,0.0]  
        kp[17] = [ 0.12,-0.08,0.0]  
        kp[18] = [-0.12,-0.08,0.0]  
        return kp

    ref_pose = make_standing_pose(NKP)

    # ── Créer une séquence avec mouvement des bras (LSF-like) ──
    target_seq = np.tile(ref_pose, (T, 1, 1)).astype(np.float32)  # (T, NKP, 3)
    t_arr = np.linspace(0, 2 * np.pi, T)

    for t in range(T):
        # Bras droit monte et descend
        target_seq[t, 3, 0] =  0.35 + 0.15 * np.cos(t_arr[t])
        target_seq[t, 3, 1] =  0.5  - 0.25 * np.sin(t_arr[t])
        target_seq[t, 4, 0] =  0.50 + 0.20 * np.cos(t_arr[t])
        target_seq[t, 4, 1] =  0.85 - 0.40 * np.sin(t_arr[t])
        # Bras gauche mouvement opposé
        target_seq[t, 6, 0] = -0.35 - 0.10 * np.cos(t_arr[t] + np.pi/4)
        target_seq[t, 6, 1] =  0.5  - 0.15 * np.sin(t_arr[t] + np.pi/4)
        target_seq[t, 7, 0] = -0.50 - 0.15 * np.cos(t_arr[t] + np.pi/4)
        target_seq[t, 7, 1] =  0.85 - 0.25 * np.sin(t_arr[t] + np.pi/4)

    # ── Prédiction = target + bruit réaliste ──
    noise = rng.normal(0, 0.06, target_seq.shape).astype(np.float32)
    pred_seq = target_seq + noise

    target_flat = target_seq.reshape(T, -1)
    pred_flat   = pred_seq.reshape(T, -1)

    # Calcul MPJPE
    mpjpe_val = float(np.linalg.norm(
        pred_seq[:, :25] - target_seq[:, :25], axis=-1).mean())
    print(f"  MPJPE = {mpjpe_val:.4f}")

    # ── 1. Grille statique ──
    print("  → Grille statique...")
    visualize(
        pred_flat   = pred_flat,
        target_flat = target_flat,
        text        = "We're going to work on an arm drill that will help",
        mpjpe       = mpjpe_val,
        dtw         = 0.407,
        pose_dim    = POSE_DIM,
        kp_dim      = 3,
        n_frames    = 6,
        save_path   = 'skeleton_compare.png',
        show        = True,
    )

    # ── 2. Heatmap ──
    print("  → Heatmap erreur...")
    visualize_error_heatmap(
        pred_flat   = pred_flat,
        target_flat = target_flat,
        pose_dim    = POSE_DIM,
        save_path   = 'heatmap.png',
        show        = True,
    )

    # ── 3. Courbe ──
    print("  → Courbe MPJPE...")
    visualize_mpjpe_curve(
        pred_flat   = pred_flat,
        target_flat = target_flat,
        text        = "arm drill",
        pose_dim    = POSE_DIM,
        save_path   = 'curve.png',
        show        = True,
    )

    print("\nTerminé! Fichiers : skeleton_compare.png, heatmap.png, curve.png")
