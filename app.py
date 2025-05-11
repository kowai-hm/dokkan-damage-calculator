import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import base64
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
    "Goku Blue": 3_840_000,
    "Vegeta Blue": 4_440_000
}

image_paths = {
    "Piccolo Daimaô": "images/daimao.png",
    "Jiren": "images/jiren.png",
    "Cell Max": "images/cell_max.png",
    "Goku Blue": "images/goku_blue.png",
    "Vegeta Blue": "images/vegeta_blue.png"
}

curve_image_path = "images/unit.png"

def encode_image(path):
    img = Image.open(path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


# ---- PARAMETRES
title = st.text_input("Titre", value=st.query_params["title"] if "title" in st.query_params else "Dégâts encaissés selon la valeur adverse")
uploaded_file = st.file_uploader("Uploader une image de perso...", type=["png", "jpg", "jpeg"])
should_compute_def = st.checkbox("DEF à calculer ?", value=int(st.query_params["shouldComputeDef"]) if "shouldComputeDef" in st.query_params else False)
if should_compute_def:
    leader = st.number_input("Leader (%)", value=int(st.query_params["leader"]) if "leader" in st.query_params else 220)
    base_def = st.number_input("DEF de base", value=int(st.query_params["baseDef"]) if "baseDef" in st.query_params else 9338)
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
    multiplicative_buff_1 = st.number_input("Boost multiplicatif 1 (%)", value=int(st.query_params["MB1"]) if "MB1" in st.query_params else 0)
    is_multiplicative_buff_1_activated = st.checkbox("Boost multiplicatif 1 activé ?", value=int(st.query_params["MB1active"]) 
                                                                                             if "MB1active" in st.query_params else False)
    multiplicative_buff_2 = st.number_input("Boost multiplicatif 2 (%)", value=int(st.query_params["MB2"]) if "MB2" in st.query_params else 0)
    is_multiplicative_buff_2_activated = st.checkbox("Boost multiplicatif 2 activé ?", value=int(st.query_params["MB2active"]) 
                                                                                             if "MB2active" in st.query_params else False)
    special_stack_value = st.number_input("Valeur de stack #1 de DEF sur la spé (%)", value=int(st.query_params["specialStackValue1"]) 
                                                                                            if "specialStackValue1" in st.query_params else 30)
    special_stack = st.number_input("Nombre de stacks #1 de DEF sur la spé", value=int(st.query_params["specialStack1"]) 
                                                                                   if "specialStack1" in st.query_params else 0)
    special_stack_value_2 = st.number_input("Valeur de stack #2 de DEF sur la spé (%)", value=int(st.query_params["specialStackValue2"]) 
                                                                                              if "specialStackValue2" in st.query_params else 30)
    special_stack_2 = st.number_input("Nombre de stacks #2 de DEF sur la spé", value=int(st.query_params["specialStack2"]) 
                                                                                   if "specialStack2" in st.query_params else 0)
    links = st.number_input("Liens (%)", value=int(st.query_params["links"]) if "links" in st.query_params else 0)
    active_skill_buff = st.number_input("Boost par active skill (%)", value=int(st.query_params["AS"]) if "AS" in st.query_params else 0)
    is_active_skill_used = st.checkbox("Active skill utilisé ?", value=int(st.query_params["ASactive"]) if "ASactive" in st.query_params else False)
else:
    defense = st.number_input("Défense", value=int(st.query_params["defense"]) if "defense" in st.query_params else 0)
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
    def compute_def(base_def=base_def, tree_completion=tree_completion, leader=leader, base=base, multiplicative_buff_1=multiplicative_buff_1, 
                    multiplicative_buff_2=multiplicative_buff_2, is_multiplicative_buff_1_activated=is_multiplicative_buff_1_activated,
                    is_multiplicative_buff_2_activated=is_multiplicative_buff_2_activated, special_stack_value=special_stack_value, 
                    special_stack=special_stack, special_stack_value_2=special_stack_value_2, special_stack_2=special_stack_2, links=links, 
                    active_skill_buff=active_skill_buff, is_active_skill_used=is_active_skill_used):
        defense = (
            (base_def + tree_completion) 
            // (1/(1 + leader*2/100))
            // (1/(1 + base/100)) 
            // (1/(1 + multiplicative_buff_1/100*is_multiplicative_buff_1_activated + multiplicative_buff_2/100*is_multiplicative_buff_2_activated))
            // (1/(1 + special_stack_value/100 * special_stack + special_stack_value_2/100 * special_stack_2)) 
            // (1/(1 + links/100))
            // (1/(1 + active_skill_buff/100 * int(is_active_skill_used)))
        )
        return defense
    defense = compute_def()

def compute_damage(x, defense=defense, damage_reduction=damage_reduction, type_multiplier=type_multiplier, guard_multiplier=guard_multiplier, 
                   type_defense_boost=type_defense_boost, is_guard_activated=is_guard_activated):
    # Type defense boost only applies if natural type advantage is applicable
    should_apply_tdb = guard_multiplier == 0.5
    if is_guard_activated:
        type_multiplier = 0.8
        guard_multiplier = 0.5
    # Average variance = 1.015
    damage = (x * 1.015 * (type_multiplier - 0.01 * type_defense_boost * int(should_apply_tdb)) * (1-damage_reduction/100) - defense) * guard_multiplier
    # Minimum damage occurs when the result of the damage equation is less than 150 ; actually, random value between 9 and 132
    return np.where(damage <= 150, 0, damage)

def find_damage_threshold(damage, defense=defense, damage_reduction=damage_reduction, type_multiplier=type_multiplier, guard_multiplier=guard_multiplier, 
                          type_defense_boost=type_defense_boost, is_guard_activated=is_guard_activated):
    # Type defense boost only applies if natural type advantage is applicable
    should_apply_tdb = guard_multiplier == 0.5
    if is_guard_activated:
        type_multiplier = 0.8
        guard_multiplier = 0.5
    threshold = (damage/guard_multiplier + defense)/(1.015 * (type_multiplier - 0.01 * type_defense_boost * int(should_apply_tdb)) * (1-damage_reduction/100))
    return threshold


nodamage_threshold = find_damage_threshold(150)  # Minimum damage occurs when the result of the damage equation is less than 150

# ---- CALCULATEUR
col1, col2 = st.columns(2)
with col1:
    st.markdown("Valeur d'ATT adverse")
    x_custom = st.number_input(
        "Valeur d'ATT adverse",
        min_value=0,
        format="%d",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("Dégâts encaissés")
    y_custom = st.number_input(
        "Dégâts encaissés",
        value=math.ceil(compute_damage(x_custom)),
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
    xaxis_title="Valeur adverse",
    yaxis_title="Dégâts encaissés",
    hovermode="x unified",
    height=600,
    xaxis_tickformat=",.0f",  # Formatage de l'axe X pour afficher les milliers avec des virgules (ou espaces)
    yaxis_tickformat=",.0f",
    xaxis=dict(
        tickfont=dict(size=22),  # Taille des labels sur l'axe X
        showgrid=True,       # Active la grille horizontale
        gridcolor='lightgray',
        gridwidth=1
    ),
    yaxis=dict(
        tickfont=dict(size=22),  # Taille des labels sur l'axe Y
        showgrid=True,       # Active la grille horizontale
        gridcolor='lightgray',
        gridwidth=1
    ),
    showlegend=False 
)

# Courbe principale
x_vals = np.linspace(
    max(0, min(nodamage_threshold, *bosses.values())-100_000), 
    max(nodamage_threshold, *bosses.values())+100_000, 
    15000)
y_vals = compute_damage(x_vals)
fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Dégâts', line=dict(color='orange')))

# Point au seuil d'annulation
fig.add_trace(go.Scatter(
        x=[nodamage_threshold], y=[0],
        mode='markers',
        marker=dict(size=8, color='red'),
        name="Seuil d'annulation",
        hovertemplate=f"Seuil d'annulation<br>Valeur adverse: {nodamage_threshold:,.0f}<extra></extra>".replace(",", " ")
    ))

# Ajout des bosses, annotations, images
encoded_images = {name: encode_image(path) for name, path in image_paths.items()}
for name, x_boss in bosses.items():
    y_boss = compute_damage(x_boss)
    
    # Ligne verticale
    fig.add_trace(go.Scatter(
        x=[x_boss, x_boss],
        y=[0, max(compute_damage(x_vals, defense=compute_def(tree_completion=min(tree)) if should_compute_def else defense))*1.1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Point cliquable
    fig.add_trace(go.Scatter(
        x=[x_boss], y=[y_boss],
        mode='markers',
        marker=dict(size=8, color='red'),
        name=name,
        hovertemplate=f"{name}<br>Valeur adverse: {x_boss:,.0f}<br>Dégâts encaissés: {y_boss:,.0f}<extra></extra>".replace(",", " ")
    ))

    # Image du boss
    fig.add_layout_image(
        dict(
            source=encoded_images[name],
            xref="x",
            yref="paper",
            x=x_boss,
            y=0.88,
            sizex=500_000,
            sizey=0.13,
            xanchor="center",
            yanchor="bottom",
            layer="above"
        )
    )

# Image associée à la courbe principale
curve_image = f"data:image/png;base64,{st.query_params["img"]}" if "img" in st.query_params else \
              encode_image(curve_image_path if not uploaded_file else uploaded_file) 
fig.add_layout_image(
    dict(
        source=curve_image,
        xref="paper",
        yref="paper",
        x=-0.07,
        y=1.18,
        sizex=0.15,
        sizey=0.15,
        xanchor="left",
        yanchor="top",
        layer="above"
    )
)

# Annotation avec la valeur de la défense sous l'image de l'unité analysée
fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0,
    y=1.18,
    xanchor="left",
    yanchor="top",
    text=f"Valeur de DEF: {defense:,.0f}".replace(",", " "),
    showarrow=False,
    font=dict(size=12, color="black"),
    align="left"
)

# ---- Ajouter l'annotation concernant la pente
x1 = 100_000_000
p1 = compute_damage(x1)
x2 = 101_000_000
p2 = compute_damage(x2)
slope = p2 - p1
fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0,
    y=1.14,
    xanchor="left",
    yanchor="top",
    text=f"Pente : + {slope:,.0f} dégâts / + {x2-x1:,.0f} ATT".replace(",", " "),
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
        y1 = compute_damage(x_vals, defense=compute_def(tree_completion=tree_completion1))
        y2 = compute_damage(x_vals, defense=compute_def(tree_completion=tree_completion2))
        
        y_min = np.minimum(y1, y2)
        y_max = np.maximum(y1, y2)

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_vals, x_vals[::-1]]),
            y=np.concatenate([y_min, y_max[::-1]]),
            fill='toself',
            fillcolor=fill_colors[i],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
    ))

    # ---- Ajouter l'annotation concernant l'impact de l'arbre sur la variation des dégâts encaissés 
    y_min = compute_damage(x_vals, defense=compute_def(tree_completion=min(tree)))
    y_max = compute_damage(x_vals, defense=compute_def(tree_completion=max(tree)))
    for ymn, ymx in zip(y_min, y_max):
        if ymn > 0 and ymx > 0:
            tree_impact = ymn - ymx
            break
    else:
        tree_impact = np.inf if y_min[-1] > 0 else 0 # Si tous les points sont à zéro
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1.10,
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
        #"img": curve_image.split(",")[1]
    }
    if should_compute_def:
        params |= {
            "leader": leader,
            "baseDef": base_def,
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
        }
    else:
        params["defense"] = defense

    query_string = urlencode(params)
    base_url = "https://dokkan-calculator.streamlit.app/"
    full_url = f"{base_url}?{query_string}"
    st.code(full_url)
    
# STREAMLIT
st.plotly_chart(fig, use_container_width=True)
