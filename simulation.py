import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. CONSTANTES PHYSIQUES (Standard ESA/NASA)
# ==========================================
MU = 398600.4418      # km^3/s^2  — Paramètre gravitationnel standard
R_EARTH = 6378.137    # km        — Rayon équatorial WGS-84
G0 = 9.80665 / 1000.0 # km/s^2   — Accélération gravitationnelle standard
ISP = 310.0           # s         — Impulsion spécifique (Green Bi-propellant)

# Paramètres mission
M_INITIAL = 500.0     # kg  — Masse initiale du Chaser
M_KIT = 24.0          # kg  — Masse du kit de déorbitisation

# Paramètres visuels (échelle de rendu)
TAILLE = 50
SCALE_EARTH = 1 / 10

# ==========================================
# 2. FONCTIONS DE MÉCANIQUE ORBITALE (test.py)
# ==========================================

def get_state_vectors(r_km, inc, omega, anomaly):
    """Calcule les vecteurs position r et vitesse v (Inertiel ECI) d'un satellite."""
    r_p = np.array([r_km * np.cos(anomaly), r_km * np.sin(anomaly), 0.0])
    v_p = np.array([
        -np.sqrt(MU / r_km) * np.sin(anomaly),
         np.sqrt(MU / r_km) * np.cos(anomaly),
         0.0
    ])
    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega),  np.cos(omega), 0],
        [0,              0,             1]
    ])
    R_inc = np.array([
        [1, 0,            0           ],
        [0, np.cos(inc), -np.sin(inc) ],
        [0, np.sin(inc),  np.cos(inc) ]
    ])
    rot = R_omega @ R_inc
    return rot @ r_p, rot @ v_p


def solve_phasing(h1, i1, o1, anom1, h2, i2, o2, anom2):
    """
    Calcule le temps d'attente optimal pour un rendez-vous de Hohmann.
    Retourne : (t_wait, t_trans, a_trans)
    """
    r1, r2 = R_EARTH + h1, R_EARTH + h2
    n1, n2 = np.sqrt(MU / r1**3), np.sqrt(MU / r2**3)

    a_trans = (r1 + r2) / 2.0
    t_trans = np.pi * np.sqrt(a_trans**3 / MU)

    alpha = n2 * t_trans              # Angle parcouru par la cible pendant le transfert
    target_lead_angle = np.pi - alpha  # Avance angulaire requise de la cible
    current_phase = anom2 - anom1

    omega_rel = n1 - n2
    if abs(omega_rel) < 1e-10:
        omega_rel = 1e-10  # Évite la division par zéro
    t_wait = (target_lead_angle - current_phase) % (2 * np.pi) / omega_rel

    return t_wait, t_trans, a_trans


def compute_dv_vectors(r_dep, r_arr, v_chaser_pre, v_target_pre, a_trans):
    """
    Calcule les Delta-V vectoriels (ΔV1 et ΔV2) pour un transfert 3D.
    h_unit est aligné avec le mouvement réel du Chaser pour éviter
    que v_p_vec soit anti-parallèle à v_chaser_pre (ΔV aberrant).
    Retourne (dv1, dv2, total_dv, h_unit).
    """
    h_vec = np.cross(r_dep, r_arr)
    norm_h = np.linalg.norm(h_vec)
    if norm_h < 1e-10:
        # Plan dégénéré — utilise le moment cinétique du Chaser directement
        h_unit = np.cross(r_dep, v_chaser_pre)
        h_norm2 = np.linalg.norm(h_unit)
        h_unit = h_unit / h_norm2 if h_norm2 > 1e-10 else np.array([0.0, 0.0, 1.0])
    else:
        h_unit = h_vec / norm_h

    # Garantir que h_unit est dans le même sens que le moment cinétique du Chaser
    # (dot positif = même demi-espace → v_p_vec dans le bon sens)
    h_chaser = np.cross(r_dep, v_chaser_pre)
    if np.dot(h_unit, h_chaser) < 0:
        h_unit = -h_unit

    v_p_mag = np.sqrt(MU * (2.0 / np.linalg.norm(r_dep) - 1.0 / a_trans))
    v_p_vec = v_p_mag * np.cross(h_unit, r_dep / np.linalg.norm(r_dep))

    v_a_mag = np.sqrt(MU * (2.0 / np.linalg.norm(r_arr) - 1.0 / a_trans))
    v_a_vec = v_a_mag * np.cross(h_unit, r_arr / np.linalg.norm(r_arr))

    dv1 = np.linalg.norm(v_p_vec - v_chaser_pre)
    dv2 = np.linalg.norm(v_target_pre - v_a_vec)
    return dv1, dv2, dv1 + dv2, h_unit


def tsiolkovsky(total_dv, m_initial=M_INITIAL):
    """Applique l'équation de Tsiolkovsky. Retourne (m_final, fuel_used)."""
    m_final = m_initial * np.exp(-total_dv / (G0 * ISP))
    return m_final, m_initial - m_final


# ==========================================
# 3. FONCTION PRINCIPALE DE VISUALISATION (last.py)
# ==========================================

def km_to_norm(r_km):
    """Convertit un rayon en km vers l'espace normalisé de rendu."""
    return (r_km / R_EARTH) * TAILLE


def norm_to_km(r_norm):
    """Inverse de km_to_norm."""
    return (r_norm / TAILLE) * R_EARTH


def plot_mission(num_debris=3, target_debris_index=0):

    # ------------------------------------------
    # Paramètres Chaser
    # ------------------------------------------
    h_chaser   = 400.0              # km
    i_chaser   = np.radians(28.0)
    o_chaser   = np.radians(0.0)
    anom_chaser = 0.0               # rad

    r1_km = R_EARTH + h_chaser
    n_chaser = np.sqrt(MU / r1_km**3)

    # Vecteurs de base de l'orbite Chaser (last.py style)
    U_chaser = np.array([np.cos(o_chaser), np.sin(o_chaser), 0.0])
    W_chaser = np.array([
        -np.sin(i_chaser) * np.sin(o_chaser),
         np.sin(i_chaser) * np.cos(o_chaser),
         np.cos(i_chaser)
    ])
    V_chaser = np.cross(W_chaser, U_chaser)

    chaser_r_norm = km_to_norm(r1_km)

    # ------------------------------------------
    # Figure Plotly
    # ------------------------------------------
    fig = go.Figure()

    # Terre (sphère bleue, last.py style)
    theta_e = np.linspace(0, 2 * np.pi, 60)
    phi_e   = np.linspace(0, np.pi, 60)
    re_disp = TAILLE * SCALE_EARTH
    x_e = re_disp * np.outer(np.cos(theta_e), np.sin(phi_e))
    y_e = re_disp * np.outer(np.sin(theta_e), np.sin(phi_e))
    z_e = re_disp * np.outer(np.ones(len(theta_e)), np.cos(phi_e))
    fig.add_trace(go.Surface(
        x=x_e, y=y_e, z=z_e,
        colorscale=[[0, 'royalblue'], [1, 'steelblue']],
        showscale=False, opacity=1.0,
        name='Earth', hoverinfo='skip'
    ))

    # Orbite Chaser
    t_circle = np.linspace(0, 2 * np.pi, 150)
    c_ox = chaser_r_norm * np.cos(t_circle) * U_chaser[0] + chaser_r_norm * np.sin(t_circle) * V_chaser[0]
    c_oy = chaser_r_norm * np.cos(t_circle) * U_chaser[1] + chaser_r_norm * np.sin(t_circle) * V_chaser[1]
    c_oz = chaser_r_norm * np.cos(t_circle) * U_chaser[2] + chaser_r_norm * np.sin(t_circle) * V_chaser[2]
    fig.add_trace(go.Scatter3d(
        x=c_ox, y=c_oy, z=c_oz,
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        opacity=0.5, name='Chaser Orbit', hoverinfo='skip'
    ))

    # Position initiale Chaser (marqueur)
    c_x0 = chaser_r_norm * np.cos(anom_chaser) * U_chaser[0] + chaser_r_norm * np.sin(anom_chaser) * V_chaser[0]
    c_y0 = chaser_r_norm * np.cos(anom_chaser) * U_chaser[1] + chaser_r_norm * np.sin(anom_chaser) * V_chaser[1]
    c_z0 = chaser_r_norm * np.cos(anom_chaser) * U_chaser[2] + chaser_r_norm * np.sin(anom_chaser) * V_chaser[2]
    fig.add_trace(go.Scatter3d(
        x=[c_x0], y=[c_y0], z=[c_z0],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='white')),
        name='Chaser',
        hovertemplate=f"Chaser<br>Altitude: {h_chaser:.0f} km<br>Vitesse: {n_chaser * r1_km:.2f} km/s<extra></extra>"
    ))

    # ------------------------------------------
    # Génération des Débris (last.py)
    # ------------------------------------------
    np.random.seed(42)
    d_min_norm = km_to_norm(R_EARTH + 200.0)
    d_max_norm = km_to_norm(R_EARTH + 2000.0)

    debris_data = []
    for i in range(num_debris):
        r_norm = np.random.uniform(d_min_norm, d_max_norm)
        r_km   = norm_to_km(r_norm)
        omega  = np.random.uniform(0, 2 * np.pi)
        inc    = np.random.uniform(0, np.pi)
        anomaly = np.random.uniform(0, 2 * np.pi)
        n_deb  = np.sqrt(MU / r_km**3)

        U = np.array([np.cos(omega), np.sin(omega), 0.0])
        W = np.array([-np.sin(inc) * np.sin(omega), np.sin(inc) * np.cos(omega), np.cos(inc)])
        V = np.cross(W, U)

        orbit_x = r_norm * np.cos(t_circle) * U[0] + r_norm * np.sin(t_circle) * V[0]
        orbit_y = r_norm * np.cos(t_circle) * U[1] + r_norm * np.sin(t_circle) * V[1]
        orbit_z = r_norm * np.cos(t_circle) * U[2] + r_norm * np.sin(t_circle) * V[2]

        deb_x = r_norm * np.cos(anomaly) * U[0] + r_norm * np.sin(anomaly) * V[0]
        deb_y = r_norm * np.cos(anomaly) * U[1] + r_norm * np.sin(anomaly) * V[1]
        deb_z = r_norm * np.cos(anomaly) * U[2] + r_norm * np.sin(anomaly) * V[2]

        size = np.random.uniform(4, 10)
        hue  = np.random.randint(0, 360)
        color = f'hsl({hue}, 80%, 65%)'

        debris_data.append({
            'r_norm': r_norm, 'r_km': r_km,
            'anomaly': anomaly, 'n': n_deb,
            'U': U, 'V': V, 'W': W,
            'inc': inc, 'omega': omega,
            'size': size, 'color': color
        })

        # Trace orbite débris
        fig.add_trace(go.Scatter3d(
            x=orbit_x, y=orbit_y, z=orbit_z, mode='lines',
            line=dict(color=color, width=1), opacity=0.5,
            showlegend=False, hoverinfo='none'
        ))
        # Trace marqueur débris initial
        fig.add_trace(go.Scatter3d(
            x=[deb_x], y=[deb_y], z=[deb_z], mode='markers',
            marker=dict(size=size, color=color, line=dict(width=1, color='white')),
            showlegend=False, name=f'Debris {i+1}',
            hovertemplate=f"Débris {i+1}<br>Alt: {r_km - R_EARTH:.0f} km<br>V: {n_deb * r_km:.2f} km/s<extra></extra>"
        ))

    # ------------------------------------------
    # Physique de test.py : solve_phasing + ΔV vectoriels
    # ------------------------------------------
    target = debris_data[target_debris_index]
    h_target   = target['r_km'] - R_EARTH
    i_target   = target['inc']
    o_target   = target['omega']
    anom_target = target['anomaly']

    t_wait, t_trans, a_trans = solve_phasing(
        h_chaser, i_chaser, o_chaser, anom_chaser,
        h_target, i_target, o_target, anom_target
    )

    # Vecteurs d'état précis aux instants clés
    n_chaser_val = np.sqrt(MU / r1_km**3)
    anom_dep   = anom_chaser + n_chaser_val * t_wait
    anom_arr   = anom_target + np.sqrt(MU / (R_EARTH + h_target)**3) * (t_wait + t_trans)

    r_dep, v_chaser_pre = get_state_vectors(r1_km,               i_chaser, o_chaser, anom_dep)
    r_arr, v_target_pre  = get_state_vectors(R_EARTH + h_target,  i_target, o_target, anom_arr)

    dv1, dv2, total_dv, h_unit = compute_dv_vectors(r_dep, r_arr, v_chaser_pre, v_target_pre, a_trans)
    m_final, fuel_used = tsiolkovsky(total_dv)

    # ------------------------------------------
    # Arc de transfert de Hohmann (rendu last.py)
    # ------------------------------------------
    r_dep_norm = km_to_norm(np.linalg.norm(r_dep))
    r_arr_norm = km_to_norm(np.linalg.norm(r_arr))
    a_trans_norm = (r_dep_norm + r_arr_norm) / 2.0
    e_trans = abs(r_arr_norm - r_dep_norm) / (r_dep_norm + r_arr_norm)
    n_trans = np.sqrt(MU / a_trans**3)   # rad/s (en km)

    # Construction de l'arc de transfert de Hohmann
    # Pour s'aligner parfaitement avec les orbites projetées en U/V :
    # On sait que le départ est dep_pt et l'arrivée est arr_pt.
    # L'arc est une ellipse dans le plan défini par dep_pt et arr_pt.
    
    # Points de manœuvre exacts dans le repère visuel (U/V)
    dep_x = chaser_r_norm * np.cos(anom_dep) * U_chaser[0] + chaser_r_norm * np.sin(anom_dep) * V_chaser[0]
    dep_y = chaser_r_norm * np.cos(anom_dep) * U_chaser[1] + chaser_r_norm * np.sin(anom_dep) * V_chaser[1]
    dep_z = chaser_r_norm * np.cos(anom_dep) * U_chaser[2] + chaser_r_norm * np.sin(anom_dep) * V_chaser[2]
    dep_pt = np.array([dep_x, dep_y, dep_z])

    arrival_anomaly = anom_arr
    arr_x = target['r_norm'] * np.cos(arrival_anomaly) * target['U'][0] + target['r_norm'] * np.sin(arrival_anomaly) * target['V'][0]
    arr_y = target['r_norm'] * np.cos(arrival_anomaly) * target['U'][1] + target['r_norm'] * np.sin(arrival_anomaly) * target['V'][1]
    arr_z = target['r_norm'] * np.cos(arrival_anomaly) * target['U'][2] + target['r_norm'] * np.sin(arrival_anomaly) * target['V'][2]
    arr_pt = np.array([arr_x, arr_y, arr_z])

    # Vecteurs de base du plan de transfert visuel
    P_vis = dep_pt / np.linalg.norm(dep_pt)
    h_vis = np.cross(dep_pt, arr_pt)
    Q_vis = np.cross(h_vis / np.linalg.norm(h_vis), P_vis)
    Q_vis = Q_vis / np.linalg.norm(Q_vis) # Normalisation par sécurité

    # Calcul de l'angle réel entre le point de départ et d'arrivée
    # Dans un transfert de Hohmann idéal on parcourt 180° (pi)
    # Dans notre rendu 3D avec différences d'inclinaison/RAAN,
    # c'est l'angle géométrique projeté qui doit dicter la fin de l'arc
    cos_theta = np.dot(dep_pt, arr_pt) / (np.linalg.norm(dep_pt) * np.linalg.norm(arr_pt))
    # On garantit que la valeur est dans le domaine de arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # L'angle doit être proche de pi (transfert semi-elliptique)
    transfer_angle = np.arccos(cos_theta)
    # L'arc de base est de la forme (0 à transfert_angle)
    # Si par hasard l'angle est inférieur à pi/2 en absolu, 
    # c'est que arr_pt et dep_pt sont presque du même coté (ce qui n'arrive pas en Hohmann)
    # Mais par construction, transfer_angle donné par arccos est l'angle direct le plus court.
    # Dans un Hohmann cross-plane, l'angle parcouru est PI.
    
    # On force le parcours à aller de 0 à transfer_angle (en se calant sur P_vis)
    # Exceptionnellement, si on va de l'orbite haute vers basse, le périapse est à l'arrivée
    if np.linalg.norm(arr_pt) >= np.linalg.norm(dep_pt):
        # r_dep < r_arr : On part du périapse (nu=0) vers l'apoapse (nu=pi)
        nu_start = 0.0
        nu_end = np.pi # Théoriquement pi
    else:
        # r_dep > r_arr : On part de l'apoapse (nu=pi) vers le périapse (nu=2pi)
        nu_start = np.pi
        nu_end = 2 * np.pi
        
    nu_arc = np.linspace(nu_start, nu_end, 80)

    arc_pts = []
    for nu in nu_arc:
        r_nu_km = a_trans * (1 - e_trans**2) / (1 + e_trans * np.cos(nu))
        r_nu_norm = km_to_norm(r_nu_km)
        
        # Le repère P_vis pointe vers dep_pt.
        # Si nu_start == 0, nu varie de 0 à pi. L'angle par rapport à P_vis est directement `nu`.
        # Si nu_start == pi, nu varie de pi à 2pi, mais P_vis pointe DÉJÀ vers le point de départ (qui correspond à nu=pi).
        # Donc l'angle par rapport à P_vis dans notre plan visuel est (nu - nu_start).
        angle = nu - nu_start
        
        # Pour forcer l'arc à vraiment se connecter au point exact (rattraper l'erreur du U/V Kepler vs Newton)
        # on interpole l'angle depuis 0 jusqu'à l'angle de transfert REEL (transfer_angle)
        fraction = angle / np.pi
        angle_visuel = fraction * transfer_angle
        
        pt = r_nu_norm * (np.cos(angle_visuel) * P_vis + np.sin(angle_visuel) * Q_vis)
        arc_pts.append(pt)
    
    # On garantit que le dernier point est EXACTEMENT arr_pt
    arc_pts[-1] = arr_pt
    arc_pts = np.array(arc_pts)

    fig.add_trace(go.Scatter3d(
        x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2],
        mode='lines',
        line=dict(color='yellow', width=5, dash='dot'),
        name=f'Hohmann Transfer → Debris {target_debris_index + 1}',
        visible='legendonly'
    ))

    # L'arc et les points de manœuvres sont maintenant parfaitement alignés

    # Rendu : étoiles avec halo, clairement distincts des sphères de débris
    fig.add_trace(go.Scatter3d(
        x=[dep_x], y=[dep_y], z=[dep_z],
        mode='markers',
        marker=dict(size=6, color='#FF6A00', symbol='diamond',
                    line=dict(width=1, color='white')),
        name='🔥 Burn 1 — Départ',
        hovertemplate=f"<b>BURN 1 — Départ</b><br>ΔV1 = {dv1 * 1000:.1f} m/s<extra></extra>"
    ))
    fig.add_trace(go.Scatter3d(
        x=[arr_x], y=[arr_y], z=[arr_z],
        mode='markers',
        marker=dict(size=6, color='#00FFB2', symbol='diamond',
                    line=dict(width=1, color='white')),
        name='🎯 Burn 2 — Rendez-vous',
        hovertemplate=f"<b>BURN 2 — Rendez-vous</b><br>ΔV2 = {dv2 * 1000:.1f} m/s<extra></extra>"
    ))

    # ------------------------------------------
    # Recalcul de la condition initiale du débris cible
    # pour que l'arrivée soit parfaite (last.py logique)
    # ------------------------------------------
    arrival_anomaly = anom_arr  # anomalie vraie à l'arrivée (calculée par solve_phasing)
    # Anomalie initiale recalée
    target['anomaly'] = arrival_anomaly - target['n'] * (t_wait + t_trans)

    # Met à jour le marqueur initial du débris cible dans la figure
    deb_name = f'Debris {target_debris_index + 1}'
    deb_new_x = target['r_norm'] * np.cos(target['anomaly']) * target['U'][0] + target['r_norm'] * np.sin(target['anomaly']) * target['V'][0]
    deb_new_y = target['r_norm'] * np.cos(target['anomaly']) * target['U'][1] + target['r_norm'] * np.sin(target['anomaly']) * target['V'][1]
    deb_new_z = target['r_norm'] * np.cos(target['anomaly']) * target['U'][2] + target['r_norm'] * np.sin(target['anomaly']) * target['V'][2]
    for tr in fig.data:
        if getattr(tr, 'name', '') == deb_name and getattr(tr, 'mode', '') == 'markers':
            tr.x = (deb_new_x,)
            tr.y = (deb_new_y,)
            tr.z = (deb_new_z,)

    # ------------------------------------------
    # Rapport console (style last.py + précision test.py)
    # ------------------------------------------
    print(f"\n==================================================")
    print(f" MISSION : INTERCEPT DEBRIS {target_debris_index + 1}")
    print(f"==================================================")
    print(f"\n[Phasing — solve_phasing()]")
    print(f"  Temps d'attente orbital : {t_wait / 3600:.2f} h")
    print(f"  Durée du transfert      : {t_trans / 60:.1f} min")
    print(f"\n[Delta-V vectoriels 3D]")
    print(f"  ΔV1 (Départ)     : {dv1 * 1000:.2f} m/s")
    print(f"  ΔV2 (Rendez-vous): {dv2 * 1000:.2f} m/s")
    print(f"  ΔV total         : {total_dv * 1000:.2f} m/s")
    print(f"\n[Propulsion — Tsiolkovsky]")
    print(f"  Carburant utilisé  : {fuel_used:.2f} kg")
    print(f"  Masse finale       : {m_final:.2f} kg")
    print(f"  Après dépôt du kit : {m_final - M_KIT:.2f} kg")
    print(f"==================================================\n")

    # ------------------------------------------
    # Animation (last.py — Kepler + Newton)
    # ------------------------------------------
    num_frames = 300
    total_sim_time = t_wait + t_trans + 3600.0
    dt = total_sim_time / num_frames

    # Indices des traces mobiles
    idx_chaser_trace = next(
        (i for i, tr in enumerate(fig.data) if getattr(tr, 'name', '') == 'Chaser'), -1
    )
    idx_debris_traces = []
    for i in range(num_debris):
        for idx, tr in enumerate(fig.data):
            if getattr(tr, 'name', '') == f'Debris {i + 1}':
                idx_debris_traces.append(idx)
    marker_traces = ([idx_chaser_trace] if idx_chaser_trace >= 0 else []) + idx_debris_traces

    frames = []
    for step in range(num_frames):
        t_sim = step * dt
        frame_data = []

        # ---- Position du Chaser ----
        if t_sim <= t_wait:
            # Phase 1 : attente sur orbite initiale
            new_anom = anom_chaser + n_chaser_val * t_sim
            cx = chaser_r_norm * np.cos(new_anom) * U_chaser[0] + chaser_r_norm * np.sin(new_anom) * V_chaser[0]
            cy = chaser_r_norm * np.cos(new_anom) * U_chaser[1] + chaser_r_norm * np.sin(new_anom) * V_chaser[1]
            cz = chaser_r_norm * np.cos(new_anom) * U_chaser[2] + chaser_r_norm * np.sin(new_anom) * V_chaser[2]

        elif t_sim <= t_wait + t_trans:
            # Phase 2 : arc de Hohmann — interpolation sur l'arc visuel (garantit la continuité)
            fraction = (t_sim - t_wait) / t_trans
            idx = min(int(fraction * len(arc_pts)), len(arc_pts) - 1)
            cx, cy, cz = arc_pts[idx]

        else:
            # Phase 3 : rendez-vous — Chaser suit le débris
            tsa = t_sim - (t_wait + t_trans)
            a_rv = arrival_anomaly + target['n'] * tsa
            cx = target['r_norm'] * np.cos(a_rv) * target['U'][0] + target['r_norm'] * np.sin(a_rv) * target['V'][0]
            cy = target['r_norm'] * np.cos(a_rv) * target['U'][1] + target['r_norm'] * np.sin(a_rv) * target['V'][1]
            cz = target['r_norm'] * np.cos(a_rv) * target['U'][2] + target['r_norm'] * np.sin(a_rv) * target['V'][2]

        frame_data.append(go.Scatter3d(x=[cx], y=[cy], z=[cz]))

        # ---- Position de chaque débris ----
        for deb in debris_data:
            a_deb = deb['anomaly'] + deb['n'] * t_sim
            dx = deb['r_norm'] * np.cos(a_deb) * deb['U'][0] + deb['r_norm'] * np.sin(a_deb) * deb['V'][0]
            dy = deb['r_norm'] * np.cos(a_deb) * deb['U'][1] + deb['r_norm'] * np.sin(a_deb) * deb['V'][1]
            dz = deb['r_norm'] * np.cos(a_deb) * deb['U'][2] + deb['r_norm'] * np.sin(a_deb) * deb['V'][2]
            frame_data.append(go.Scatter3d(x=[dx], y=[dy], z=[dz]))

        frames.append(go.Frame(data=frame_data, traces=marker_traces, name=str(step)))

    fig.frames = frames

    # ------------------------------------------
    # Layout (last.py style)
    # ------------------------------------------
    max_r = km_to_norm(R_EARTH + 2000.0) * 1.1

    # Visibilité : masque l'arc par défaut
    vis_with_arc    = [True] * len(fig.data)
    vis_without_arc = [True] * len(fig.data)
    for idx, tr in enumerate(fig.data):
        if (tr.name or '').startswith('Hohmann Transfer'):
            vis_without_arc[idx] = False

    fig.update_layout(
        title=dict(
            text="3D Mission de Rendez-Vous Orbital — Mécanique Rigoureuse",
            font=dict(color='white', size=22, family='Arial'),
            x=0.5, y=0.95
        ),
        paper_bgcolor='black',
        scene=dict(
            xaxis=dict(visible=False, range=[-max_r, max_r]),
            yaxis=dict(visible=False, range=[-max_r, max_r]),
            zaxis=dict(visible=False, range=[-max_r, max_r]),
            bgcolor='black',
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                bgcolor='white',
                font=dict(color='black', size=13),
                y=0.10, x=0.10, xanchor="right", yanchor="top",
                buttons=[
                    dict(
                        label="► Play / ❚❚ Pause",
                        method="animate",
                        args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True, transition=dict(duration=0))],
                        args2=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate", transition=dict(duration=0))]
                    )
                ]
            ),
            dict(
                type="buttons",
                showactive=True,
                bgcolor='white',
                font=dict(color='black', size=13),
                y=0.20, x=0.10, xanchor="right", yanchor="top",
                buttons=[
                    dict(
                        label="Afficher l'arc de transfert",
                        method="restyle", # Restyle au lieu de update pour ne pas reset la caméra
                        args=[{"visible": vis_with_arc}],
                        args2=[{"visible": vis_without_arc}]
                    )
                ]
            )
        ]
    )

    # ------------------------------------------
    # Export HTML (last.py)
    # ------------------------------------------
    output_filename = "mission_rendez_vous.html"
    print(f"Génération du fichier interactif : {output_filename} ...")
    fig.write_html(output_filename, auto_open=True)
    print("Done ! La mission s'ouvre dans votre navigateur.")


# ==========================================
# POINT D'ENTRÉE
# ==========================================
if __name__ == "__main__":
    plot_mission(num_debris=3, target_debris_index=0)
