import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import base64
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlencode


st.set_page_config(layout="wide")


# ---- DATA & IMAGES
guard = {
    "Même classe & désavantage type":       {"type_multiplier": 1.25, "guard_multiplier": 1},
    "Classe opposée & désavantage type":    {"type_multiplier": 1.5,  "guard_multiplier": 1},
    "Même classe & neutre en type":         {"type_multiplier": 1.0,  "guard_multiplier": 1},
    "Classe opposée & neutre en type":      {"type_multiplier": 1.15, "guard_multiplier": 1},
    "Même classe & avantage type":          {"type_multiplier": 0.9,  "guard_multiplier": 0.5},
    "Classe opposée & avantage type":       {"type_multiplier": 1.0,  "guard_multiplier": 0.5},
}

trees = {
    "TEC": [2000, 3700, 4000, 4310, 5000],
    "AGI": [2000, 4100, 4400, 4710, 5400],
    "PUI": [2000, 3300, 3600, 3910, 4600],
    "END": [2000, 3300, 3600, 3910, 4600],
    "INT": [2000, 3700, 4000, 4310, 5000],
}

bosses = {
    "Piccolo Daimaô": 4_800_000,
    "Jiren": 5_740_000,
    "Cell Max": 6_562_500,
    #"Goku Blue": 4_560_000,
    #"Vegeta Blue": 4_440_000,
    "Trunks SoH": 7_700_000,
    #"Gogeta SSJ4": 9_360_000,
    "Goku SSJ4 (DAIMA)": 12_000_000,
    "Vegeta SSJ4": 16_100_000,
    "Omega Shenron": 10_500_000,
}

image_paths = {
    "Piccolo Daimaô": "images/daimao.png",
    "Jiren": "images/jiren.png",
    "Cell Max": "images/cell_max.png",
    #"Goku Blue": "images/goku_blue.png",
    #"Vegeta Blue": "images/vegeta_blue.png",
    "Trunks SoH": "images/trunks.png",
    #"Gogeta SSJ4": "images/gogeta_ssj4.png",
    "Goku SSJ4 (DAIMA)": "images/goku_ssj4_daima.png",
    "Vegeta SSJ4": "images/vegeta_ssj4.png",
    "Omega Shenron": "images/omega_shenron.png",
}

curve_image_path = "images/unit.png"

def encode_image(path):
    img = Image.open(path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


# ---- PARAMETRES
title = st.text_input("Titre", value=st.query_params["title"] if "title" in st.query_params else "Dégâts encaissés selon les PV restants")
uploaded_file = st.file_uploader("Uploader une image de perso...", type=["png", "jpg", "jpeg"])
url_file = st.text_input("URL pour une image de perso...", value=st.query_params["img"] if "img" in st.query_params else "")
should_compute_def = st.checkbox("DÉF à calculer ?", value=int(st.query_params["shouldComputeDef"]) if "shouldComputeDef" in st.query_params else False)
if should_compute_def:
    leader = st.number_input("Leader (%)", value=int(st.query_params["leader"]) if "leader" in st.query_params else 220)
    base_def = st.number_input("DÉF de base", value=int(st.query_params["baseDef"]) if "baseDef" in st.query_params else 9338)
    equips = st.number_input("Équipements de DÉF", value=int(st.query_params["equips"]) if "equips" in st.query_params else 0)
    type_selection = st.selectbox("Type", list(trees.keys()), index=int(st.query_params["typeSelection"]) if "typeSelection" in st.query_params else 0)
    tree = trees[type_selection]
    rank_s_selection = st.checkbox("Rang S ?", value=int(st.query_params["rankS"]) if "rankS" in st.query_params else False)
    if rank_s_selection:
        tree = [int(v*1.4) for v in tree]
    free_unit_selection = st.checkbox("F2P ?", value=int(st.query_params["f2p"]) if "f2p" in st.query_params else False)
    if free_unit_selection:
        tree = [int(v*0.6) for v in tree]
    tree_completion_selection = st.selectbox("Complétion de l'arbre", tree, index=int(st.query_params["treeCompletionSelection"]) 
                                                                                  if "treeCompletionSelection" in st.query_params else 0)
    tree_completion = tree_completion_selection
    base = st.number_input("Base (%)", value=int(st.query_params["base"]) if "base" in st.query_params else 0)
    support = st.number_input("Support non multiplicatif (%)", value=int(st.query_params["support"]) if "support" in st.query_params else 0)
    multiplicative_buff_1 = st.number_input("Boost multiplicatif 1 (%)", value=int(st.query_params["MB1"]) if "MB1" in st.query_params else 0)
    is_multiplicative_buff_1_activated = st.checkbox("Boost multiplicatif 1 activé ?", value=int(st.query_params["MB1active"]) 
                                                                                             if "MB1active" in st.query_params else False)
    multiplicative_buff_2 = st.number_input("Boost multiplicatif 2 (%)", value=int(st.query_params["MB2"]) if "MB2" in st.query_params else 0)
    is_multiplicative_buff_2_activated = st.checkbox("Boost multiplicatif 2 activé ?", value=int(st.query_params["MB2active"]) 
                                                                                             if "MB2active" in st.query_params else False)
    special_stack_value = st.number_input("Valeur de stack #1 de DÉF sur la spé (%)", value=int(st.query_params["specialStackValue1"]) 
                                                                                            if "specialStackValue1" in st.query_params else 30)
    special_stack = st.number_input("Nombre de stacks #1 de DÉF sur la spé", value=int(st.query_params["specialStack1"]) 
                                                                                   if "specialStack1" in st.query_params else 0)
    special_stack_value_2 = st.number_input("Valeur de stack #2 de DÉF sur la spé (%)", value=int(st.query_params["specialStackValue2"]) 
                                                                                              if "specialStackValue2" in st.query_params else 30)
    special_stack_2 = st.number_input("Nombre de stacks #2 de DÉF sur la spé", value=int(st.query_params["specialStack2"]) 
                                                                                   if "specialStack2" in st.query_params else 0)
    links = st.number_input("Liens (%)", value=int(st.query_params["links"]) if "links" in st.query_params else 0)
    active_skill_buff = st.number_input("Boost par active skill (%)", value=int(st.query_params["AS"]) if "AS" in st.query_params else 0)
    is_active_skill_used = st.checkbox("Active skill utilisé ?", value=int(st.query_params["ASactive"]) if "ASactive" in st.query_params else False)
    item = st.number_input("Boost par item (%)", value=int(st.query_params["item"]) if "item" in st.query_params else 0)
    is_item_active = st.checkbox("Item actif ?", value=int(st.query_params["itemActive"]) if "itemActive" in st.query_params else False)
else:
    defense = st.number_input("Défense", value=int(st.query_params["defense"]) if "defense" in st.query_params else 0)
    defense_at_full_pv = defense
hp_variable_boost = st.number_input("Boost de DÉF selon les PV — max à 100% PV (%)", value=int(st.query_params["hpVariableBoost"]) if "hpVariableBoost" in st.query_params else 0)
att_fixe = st.number_input("Valeur d'ATT adverse (fixe)", min_value=0, value=int(st.query_params["attFixe"]) if "attFixe" in st.query_params else 10_000_000, format="%d")
damage_reduction = st.number_input("Réduction de dégâts (%)", value=int(st.query_params["damageReduction"]) if "damageReduction" in st.query_params else 0)
type_defense_boost = st.select_slider("Défense de type boostée", options=[5, 6, 7, 8, 10, 15], value=int(st.query_params["defenseTypeBoost"])
                                                                                                     if "defenseTypeBoost" in st.query_params else 5)
guard_selection = st.selectbox("Multiplicateur de type", list(guard.keys()), index=int(st.query_params["guardSelection"]) 
                                                                                   if "guardSelection" in st.query_params else 3)
is_guard_activated = st.checkbox("Garde passive activée ?", value=int(st.query_params["guard"]) if "guard" in st.query_params else False)
type_multiplier = guard[guard_selection]["type_multiplier"]
guard_multiplier = guard[guard_selection]["guard_multiplier"]

# ---- CALCUL
if should_compute_def:
    def compute_def(base_def=base_def, equips=equips, tree_completion=tree_completion, leader=leader, base=base, multiplicative_buff_1=multiplicative_buff_1, 
                    multiplicative_buff_2=multiplicative_buff_2, is_multiplicative_buff_1_activated=is_multiplicative_buff_1_activated,
                    is_multiplicative_buff_2_activated=is_multiplicative_buff_2_activated, special_stack_value=special_stack_value, 
                    special_stack=special_stack, special_stack_value_2=special_stack_value_2, special_stack_2=special_stack_2, links=links, 
                    active_skill_buff=active_skill_buff, is_active_skill_used=is_active_skill_used, support=support, item=item, is_item_active=is_item_active,
                    hp_variable_boost=hp_variable_boost, hp_boost_x=1.0):
        defense = (
            (base_def + equips + tree_completion) 
            // (1/(1 + leader*2/100))
            // (1/(1 + base/100 + hp_variable_boost*hp_boost_x/100 + support/100)) 
            // (1/(1 + item/100 * int(is_item_active)))
            // (1/(1 + active_skill_buff/100 * int(is_active_skill_used)))
            // (1/(1 + links/100))
            // (1/(1 + multiplicative_buff_1/100*is_multiplicative_buff_1_activated + multiplicative_buff_2/100*is_multiplicative_buff_2_activated))
            // (1/(1 + special_stack_value/100 * special_stack + special_stack_value_2/100 * special_stack_2)) 
        )
        return defense
    defense = compute_def(hp_boost_x=1.0)

# defense_at_full_pv used as the reference value for annotations and non-compute_def mode
defense_at_full_pv = defense

def compute_damage(hp_x, defense=None, damage_reduction=damage_reduction, type_multiplier=type_multiplier, guard_multiplier=guard_multiplier, 
                   type_defense_boost=type_defense_boost, is_guard_activated=is_guard_activated):
    # hp_x : fraction de PV restants (1.0 = 100% PV, 0.0 = 0% PV)
    # Si defense n'est pas fourni, on le recalcule avec le boost PV courant
    if defense is None:
        if should_compute_def:
            defense = compute_def(hp_boost_x=hp_x)
        else:
            defense = defense_at_full_pv
    # Type defense boost only applies if natural type advantage is applicable
    should_apply_tdb = guard_multiplier == 0.5
    if is_guard_activated:
        type_multiplier = 0.8
        guard_multiplier = 0.5
    # Average variance = 1.015
    damage = (att_fixe * 1.015 * (type_multiplier - 0.01 * type_defense_boost * int(should_apply_tdb)) * (1-damage_reduction/100) - defense) * guard_multiplier
    # Minimum damage occurs when the result of the damage equation is less than 150 ; actually, random value between 9 and 132
    return np.where(damage <= 150, 0, damage)


# ---- CALCULATEUR
col1, col2 = st.columns(2)
with col1:
    st.markdown("PV restants (%)")
    x_custom = st.number_input(
        "PV restants (%)",
        min_value=0,
        max_value=100,
        value=100,
        format="%d",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("Dégâts encaissés")
    y_custom = st.number_input(
        "Dégâts encaissés",
        value=math.ceil(float(compute_damage(x_custom))),
        format="%d",
        disabled=True,
        label_visibility="collapsed"
    )

# ---- GRAPHIQUE
fig = go.Figure()

# Mise en page responsive
fig.update_layout(
    title=dict(
        text=title,
        x=0.5,
        y=0.99,
        xanchor="center"
    ),
    xaxis_title="PV restants (%)",
    yaxis_title="Dégâts encaissés",
    hovermode="x unified",
    height=600,
    xaxis_tickformat=".0f",
    yaxis_tickformat=",.0f",
    xaxis=dict(
        tickfont=dict(size=22),
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        ticksuffix="%",
        autorange="reversed",  # 100% à gauche, 0% à droite
    ),
    yaxis=dict(
        tickfont=dict(size=22),
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1
    ),
    showlegend=False 
)

# Courbe principale : x = % PV de 0.0 à 1.0, affiché de 100 à 0
x_vals = np.linspace(0.0, 1.0, 5000)
y_vals = compute_damage(x_vals)
fig.add_trace(go.Scatter(x=x_vals * 100, y=y_vals, mode='lines', name='Dégâts', line=dict(color='orange')))

# Point au seuil d'annulation (% PV où les dégâts deviennent nuls)
# On cherche le hp_x tel que compute_damage(hp_x) = 0, par recherche binaire
if float(compute_damage(1.0)) <= 0:
    nodamage_pv_pct = 100.0  # Déjà annulé à 100% PV
elif float(compute_damage(0.0)) > 0:
    nodamage_pv_pct = None   # Jamais annulé
else:
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if float(compute_damage(mid)) <= 0:
            lo = mid
        else:
            hi = mid
    nodamage_pv_pct = hi * 100

if nodamage_pv_pct is not None and 0 <= nodamage_pv_pct <= 100:
    fig.add_trace(go.Scatter(
        x=[nodamage_pv_pct], y=[0],
        mode='markers',
        marker=dict(size=8, color='red'),
        name="Seuil d'annulation",
        hovertemplate=f"Seuil d'annulation<br>PV restants: {nodamage_pv_pct:.1f}%<extra></extra>"
    ))

# Image associée à la courbe principale
if url_file:
    curve_image = f"data:image/png;base64,{base64.b64encode(requests.get(url_file).content).decode('utf-8')}"
elif uploaded_file:
    curve_image = encode_image(uploaded_file)
else:
    curve_image = encode_image(curve_image_path) 
fig.add_layout_image(
    dict(
        source=curve_image,
        xref="paper",
        yref="paper",
        #x=-0.07,
        #y=1.18,
        x=-0.05,
        y=1.12,
        sizex=0.15,
        sizey=0.15,
        xanchor="left",
        yanchor="top",
        layer="above"
    )
)

# Annotation avec la valeur de la défense à 100% PV
fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0.05,
    y=1.12,
    xanchor="left",
    yanchor="top",
    text=f"Valeur de DEF (100% PV): {defense_at_full_pv:,.0f}".replace(",", " "),
    showarrow=False,
    font=dict(size=12, color="black"),
    align="left"
)

# ---- Ajouter l'annotation concernant la pente (dégâts / % PV perdu)
# On compare à 100% PV et 99% PV
p1 = float(compute_damage(1.00))
p2 = float(compute_damage(0.99))
slope = p2 - p1  # augmentation des dégâts pour 1% de PV perdu
fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0.05,
    y=1.10,
    xanchor="left",
    yanchor="top",
    text=f"Pente : + {slope:,.0f} dégâts / 1% PV perdu".replace(",", " "),
    showarrow=False,
    font=dict(size=12, color="black"),
    align="left"
)

if should_compute_def:
    # ---- Dégradés de couleur selon la valeur de l'arbre pour représenter son impact
    fill_colors = [
        'rgba(230, 159, 0, 0.3)',
        'rgba(86, 180, 233, 0.3)',
        'rgba(0, 158, 115, 0.3)',
        'rgba(204, 121, 167, 0.3)'
    ]

    for i, (tree_completion1, tree_completion2) in enumerate(zip(tree, tree[1:])):
        y1 = np.array([float(compute_damage(hx, defense=compute_def(tree_completion=tree_completion1, hp_boost_x=hx))) for hx in x_vals])
        y2 = np.array([float(compute_damage(hx, defense=compute_def(tree_completion=tree_completion2, hp_boost_x=hx))) for hx in x_vals])
        
        y_min = np.minimum(y1, y2)
        y_max = np.maximum(y1, y2)

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_vals * 100, (x_vals * 100)[::-1]]),
            y=np.concatenate([y_min, y_max[::-1]]),
            fill='toself',
            fillcolor=fill_colors[i],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
    ))

    # ---- Ajouter l'annotation concernant l'impact de l'arbre sur la variation des dégâts encaissés 
    y_min_arr = np.array([float(compute_damage(hx, defense=compute_def(tree_completion=min(tree), hp_boost_x=hx))) for hx in x_vals])
    y_max_arr = np.array([float(compute_damage(hx, defense=compute_def(tree_completion=max(tree), hp_boost_x=hx))) for hx in x_vals])
    for ymn, ymx in zip(y_min_arr, y_max_arr):
        if ymn > 0 and ymx > 0:
            tree_impact = ymn - ymx
            break
    else:
        tree_impact = np.inf if y_min_arr[-1] > 0 else 0  # Si tous les points sont à zéro
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.05,
        y=1.08,
        xanchor="left",
        yanchor="top",
        text=f"Impact maximal de l'arbre : {tree_impact:,.0f}".replace(",", " "),
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left"
    )

# BOUTON PARTAGE DE FEUILLE DE CALCUL PRECISE
if st.button("🔗 Partager feuille de calcul"):
    params = {
        "title": title,
        "shouldComputeDef": int(should_compute_def),
        "damageReduction": damage_reduction,
        "defenseTypeBoost": type_defense_boost,
        "guardSelection": list(guard.keys()).index(guard_selection),
        "guard": int(is_guard_activated),
        "img": url_file,
        "attFixe": att_fixe,
        "hpVariableBoost": hp_variable_boost,
    }
    if should_compute_def:
        params |= {
            "leader": leader,
            "baseDef": base_def,
            "equips": equips,
            "typeSelection": list(trees.keys()).index(type_selection),
            "rankS": int(rank_s_selection),
            "f2p": int(free_unit_selection),
            "treeCompletionSelection": tree.index(tree_completion_selection),
            "base": base,
            "MB1": multiplicative_buff_1,
            "MB1active": int(is_multiplicative_buff_1_activated),
            "MB2": multiplicative_buff_2,
            "MB2active": int(is_multiplicative_buff_2_activated),
            "specialStackValue1": special_stack_value,
            "specialStack1": special_stack,
            "specialStackValue2": special_stack_value_2,
            "specialStack2": special_stack_2,
            "links": links,
            "AS": active_skill_buff,
            "ASactive": int(is_active_skill_used),
            "support": support,
            "item": item,
            "itemActive": int(is_item_active)
        }
    else:
        params["defense"] = defense_at_full_pv

    query_string = urlencode(params)
    base_url = "https://dokkan-calculator.streamlit.app/"
    #base_url = "http://192.168.0.102:8501/"
    full_url = f"{base_url}?{query_string}"
    st.code(full_url)
    
# STREAMLIT
st.plotly_chart(fig, use_container_width=True)
