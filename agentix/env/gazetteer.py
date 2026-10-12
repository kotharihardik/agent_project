"""Synthetic gazetteer: streets, landmarks and localities on top of the pipe network.

Pipes in towns follow streets, so streets = chains of pipes between intersections, merged when they continue straight.
Landmarks are placed where people are (demand as a footfall proxy). Several naming 'skins' share one geometry so the
report reader learns to extract names, not to memorise one town.
"""
from __future__ import annotations

import json
import math

import networkx as nx
import numpy as np

from .network import NetworkInfo

STREET_POOLS = {
    "ltown": ["MG Road", "Station Road", "Temple Street", "Lake View Road", "Gandhi Road", "Nehru Street",
              "Market Road", "School Lane", "Hill Road", "Church Street", "Park Avenue", "Canal Road",
              "Old Mill Road", "Bazaar Street", "Tank Bund Road", "Ring Road", "College Road", "Hospital Road",
              "Water Tank Road", "Bus Stand Road", "Shivaji Marg", "Tilak Road", "Patel Street", "Ambedkar Road",
              "Netaji Road", "Rajiv Gandhi Road", "Indira Nagar Main Road", "Sai Baba Road", "Mosque Road",
              "Post Office Road", "Railway Feeder Road", "Ganesh Gully", "Kamala Street", "Rose Lane",
              "Mango Lane", "Banyan Street", "Lotus Lane", "Jasmine Street", "Sunrise Road", "Green Park Road"],
    "town2": ["Bank Street", "Gurudwara Road", "Mill Road", "Fort Road", "Kings Road", "Queens Road",
              "Clock Tower Road", "Mall Road", "Cantonment Road", "Victoria Road", "Hanuman Gully", "Kalyan Marg",
              "Sardar Patel Road", "Subhash Road", "Laxmi Street", "Saraswati Road", "Durga Lane", "Krishna Road",
              "Godavari Street", "Kaveri Road", "Ganga Road", "Yamuna Lane", "Narmada Street", "Tapi Road",
              "Coconut Grove Road", "Neem Street", "Peepal Lane", "Teak Road", "Sandal Street", "Cotton Road",
              "Silk Street", "Jute Lane", "Spice Road", "Tea Garden Road", "Coffee Street", "Sugar Mill Road",
              "Salt Pan Road", "Harbour Road", "Lighthouse Road", "Beach Road"],
    "town3": ["Main Road", "1st Cross", "2nd Cross", "3rd Cross", "4th Cross", "5th Cross", "6th Cross",
              "1st Main", "2nd Main", "3rd Main", "4th Main", "5th Main", "8th Cross", "9th Cross", "10th Cross",
              "11th Main", "12th Main", "Outer Ring Road", "Inner Ring Road", "Service Road", "BDA Road",
              "KHB Colony Road", "Layout Main Road", "Water Works Road", "Sewage Farm Road", "Industrial Road",
              "Tannery Road", "Brick Kiln Road", "Quarry Road", "Dairy Road", "Mandi Road", "Bypass Road",
              "Highway Road", "Toll Gate Road", "Check Post Road", "Bus Depot Road", "Godown Street",
              "Weavers Colony Road", "Potters Street", "Fishermen Lane"],
    "town4": ["Ashok Marg", "Vikas Marg", "Shanti Path", "Seva Marg", "Pragati Road", "Unity Road",
              "Freedom Street", "Republic Road", "Liberty Lane", "Harmony Street", "Prosperity Road", "Hope Street",
              "Faith Lane", "Dharma Road", "Karma Street", "Moksha Lane", "Surya Road", "Chandra Street",
              "Tara Lane", "Akash Road", "Prithvi Street", "Agni Lane", "Vayu Road", "Jal Street", "Van Lane",
              "Pahad Road", "Nadi Street", "Sagar Road", "Jheel Lane", "Bagh Road", "Kheti Street", "Gaon Road",
              "Shehar Street", "Mohalla Lane", "Basti Road", "Chowk Street", "Gali Lane", "Naka Road",
              "Phata Street", "Adda Road"],
}
LOCALITY_POOLS = {
    "ltown": ["Gandhi Nagar", "Shanti Colony", "Lake View", "Hill Top", "Old Town", "Market Area", "Station Area",
              "Green Park", "Sai Nagar", "Indira Colony", "Teachers Colony", "Tank Bund Area", "Canal Side",
              "Rose Garden", "New Extension", "Low Town"],
    "town2": ["Civil Lines", "Cantonment", "Fort Area", "Clock Tower", "Mill Colony", "Harbour Side", "Bank Colony",
              "Gurudwara Area", "Mall Area", "Kings Park", "Queens Colony", "Railway Colony", "Coconut Grove",
              "Tea Garden", "Spice Market", "Beach Side"],
    "town3": ["1st Stage", "2nd Stage", "3rd Phase", "4th Block", "5th Block", "KHB Colony", "BDA Layout",
              "Industrial Area", "Weavers Colony", "Potters Colony", "Mandi Area", "Bus Depot Area",
              "Brick Kiln Area", "Quarry Side", "Dairy Colony", "Toll Gate Area"],
    "town4": ["Ashok Vihar", "Vikas Puri", "Shanti Kunj", "Seva Nagar", "Pragati Vihar", "Unity Enclave",
              "Azad Nagar", "Republic Colony", "Surya Nagar", "Chandra Vihar", "Tara Puram", "Akash Ganga",
              "Sagar Vihar", "Bagh Colony", "Gaon Basti", "Chowk Area"],
}
# (type, count, aliases-template, footfall weight, prefers commercial demand)
LANDMARK_TYPES = [
    ("school", 7, ["govt school", "sarkari school", "school"], 3.0, False),
    ("temple", 7, ["mandir", "temple"], 2.0, False),
    ("mosque", 3, ["masjid", "mosque"], 2.0, False),
    ("church", 2, ["church"], 1.5, False),
    ("hospital", 2, ["hospital", "aspatal"], 2.5, True),
    ("clinic", 3, ["clinic", "dawakhana"], 1.5, True),
    ("market", 3, ["market", "bazaar", "mandi"], 5.0, True),
    ("bus_stop", 6, ["bus stop", "bus stand"], 3.0, True),
    ("petrol_pump", 2, ["petrol pump", "petrol bunk"], 2.0, True),
    ("atm", 4, ["ATM", "bank"], 2.0, True),
    ("park", 4, ["park", "garden"], 1.5, False),
    ("police_station", 1, ["police station", "thana"], 1.5, True),
    ("post_office", 1, ["post office"], 1.2, True),
    ("community_hall", 2, ["community hall", "function hall"], 1.2, False),
    ("kirana_shop", 8, ["kirana", "general store", "shop"], 2.0, True),
    ("water_tank", 1, ["water tank", "tanki"], 1.0, False),
]
LANDMARK_NAMES = {
    "school": ["Government High School", "St. Mary's School", "Vidya Mandir School", "Little Flower School",
               "Zilla Parishad School", "Sunrise Public School", "Model School", "Saraswati Vidyalaya"],
    "temple": ["Hanuman Temple", "Shiva Temple", "Ganesh Temple", "Durga Temple", "Sai Baba Temple",
               "Venkateswara Temple", "Ram Mandir", "Ayyappa Temple"],
    "mosque": ["Jama Masjid", "Noor Masjid", "Mecca Masjid"],
    "church": ["CSI Church", "St. Joseph's Church"],
    "hospital": ["Government Hospital", "City Hospital"],
    "clinic": ["Apollo Clinic", "Sri Sai Clinic", "Family Clinic"],
    "market": ["Vegetable Market", "Main Bazaar", "Fish Market"],
    "bus_stop": ["Main Bus Stand", "Hospital Bus Stop", "School Bus Stop", "Temple Bus Stop", "Market Bus Stop",
                 "Colony Bus Stop"],
    "petrol_pump": ["Indian Oil Petrol Pump", "HP Petrol Bunk"],
    "atm": ["SBI ATM", "HDFC ATM", "Canara Bank", "Post Office ATM"],
    "park": ["Gandhi Park", "Children's Park", "Rose Garden", "Lake Park"],
    "police_station": ["Police Station"],
    "post_office": ["Head Post Office"],
    "community_hall": ["Community Hall", "Kalyana Mandapam"],
    "kirana_shop": ["Lakshmi Kirana", "Balaji General Store", "Raju Kirana", "Sri Ram Stores", "Ganesh Kirana",
                    "New Krishna Stores", "Om Sai Kirana", "Annapurna Stores"],
    "water_tank": ["Overhead Water Tank"],
}


def _bearing(xy_a, xy_b):
    return math.degrees(math.atan2(xy_b[1] - xy_a[1], xy_b[0] - xy_a[0])) % 180


def build_streets(net: NetworkInfo, max_len_m: float = 700.0, turn_deg: float = 30.0):
    """Chains of degree-2 nodes between intersections, then merge straight continuations."""
    g = net.pipe_graph
    xy = {n: net.wn.get_node(n).coordinates for n in g.nodes}
    deg = dict(g.degree())
    seen_edges, chains = set(), []
    for u in g.nodes:
        if deg[u] == 2:
            continue
        for v in g.neighbors(u):
            if (u, v) in seen_edges:
                continue
            chain = [u, v]
            seen_edges.update({(u, v), (v, u)})
            while deg[chain[-1]] == 2:
                nxt = [w for w in g.neighbors(chain[-1]) if w != chain[-2]]
                if not nxt:
                    break
                seen_edges.update({(chain[-1], nxt[0]), (nxt[0], chain[-1])})
                chain.append(nxt[0])
                if chain[-1] == u:
                    break
            chains.append(chain)
    # loops made only of degree-2 nodes
    for u, v in g.edges:
        if (u, v) not in seen_edges:
            chains.append([u, v])
            seen_edges.update({(u, v), (v, u)})

    def length(ch):
        return sum(g[a][b]["length"] for a, b in zip(ch, ch[1:]))

    # merge chains that continue straight through an intersection
    chains = [list(c) for c in chains]
    merged = True
    while merged:
        merged = False
        ends = {}
        for i, c in enumerate(chains):
            ends.setdefault(c[0], []).append((i, 0))
            ends.setdefault(c[-1], []).append((i, -1))
        for node, lst in ends.items():
            if len(lst) < 2:
                continue
            best = None
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    (i, ei), (k, ek) = lst[a], lst[b]
                    if i == k:
                        continue
                    ci, ck = chains[i], chains[k]
                    pi = ci[1] if ei == 0 else ci[-2]
                    pk = ck[1] if ek == 0 else ck[-2]
                    d = abs(_bearing(xy[pi], xy[node]) - _bearing(xy[node], xy[pk]))
                    d = min(d, 180 - d)
                    if d < turn_deg and length(ci) + length(ck) < max_len_m and (best is None or d < best[0]):
                        best = (d, i, ei, k, ek)
            if best:
                _, i, ei, k, ek = best
                ci = chains[i] if ei == -1 else chains[i][::-1]
                ck = chains[k] if ek == 0 else chains[k][::-1]
                chains[i] = ci + ck[1:]
                chains.pop(k)
                merged = True
                break
    chains.sort(key=length, reverse=True)
    return chains, [length(c) for c in chains]


def build_gazetteer(net: NetworkInfo, seed: int = 0, n_localities: int = 14, min_street_m: float = 120.0):
    rng = np.random.default_rng(seed)
    chains, lengths = build_streets(net)
    J = net.index
    # --- localities: k-means on coordinates, within-zone ---
    xy = net.xy
    k = n_localities
    centers = xy[rng.choice(len(xy), k, replace=False)]
    for _ in range(30):
        lab = np.argmin(((xy[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        centers = np.array([xy[lab == c].mean(0) if np.any(lab == c) else centers[c] for c in range(k)])
    # order localities by elevation so names like "Hill Top" land high (cosmetic)
    loc_elev = np.array([net.elevation[lab == c].mean() for c in range(k)])
    loc_order = np.argsort(-loc_elev)
    locality_of = {net.junctions[i]: int(np.where(loc_order == lab[i])[0][0]) for i in range(len(xy))}

    # --- streets: long chains get names; short stubs inherit "lane" names of their parent area ---
    streets = []
    for ci, (ch, L) in enumerate(zip(chains, lengths)):
        nodes = [n for n in ch if n in J]
        if not nodes:
            continue
        pipes = [net.pipe_graph[a][b]["link"] for a, b in zip(ch, ch[1:])]
        streets.append({"sid": f"ST{ci:03d}", "nodes": nodes, "pipes": pipes, "length_m": round(L, 1),
                        "named": L >= min_street_m})
    # --- landmarks ---
    demand = net.base_demand
    comm = np.array([sum(d.base_value for d in net.wn.get_node(n).demand_timeseries_list
                         if d.pattern_name == "P-Commercial") for n in net.junctions])
    weight_res = demand / demand.sum()
    weight_com = (comm + 1e-9) / (comm + 1e-9).sum()
    chosen_nodes: list[int] = []
    landmarks = []
    dist = net.pipe_distance
    for ltype, count, aliases, footfall, prefers_comm in LANDMARK_TYPES:
        w = 0.5 * weight_res + 0.5 * (weight_com if prefers_comm else weight_res)
        names = LANDMARK_NAMES[ltype]
        for i in range(count):
            for _ in range(200):
                j = int(rng.choice(len(w), p=w))
                if all(dist[j, c] > 120 for c in chosen_nodes):
                    break
            chosen_nodes.append(j)
            landmarks.append({"lid": f"LM{len(landmarks):03d}", "type": ltype, "name_idx": i % len(names),
                              "node": net.junctions[j], "footfall": footfall, "alias_words": aliases})
    geometry = {"localities": {net.junctions[i]: locality_of[net.junctions[i]] for i in range(len(xy))},
                "streets": streets, "landmarks": landmarks, "n_localities": k}
    return geometry


def _stable_hash(s: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(s))


def skin(geometry: dict, town_id: str, seed: int = 0) -> dict:
    """Give names to a geometry with one naming pool -> gazetteer in the docs/03 format (+ geometry for our side)."""
    rng = np.random.default_rng(seed + _stable_hash(town_id) % 1000)
    street_pool = list(STREET_POOLS[town_id])
    rng.shuffle(street_pool)
    loc_pool = LOCALITY_POOLS[town_id]
    prefix = {"ltown": "LT", "town2": "T2", "town3": "T3", "town4": "T4"}[town_id]
    places, node_street = [], {}
    used = 0
    cross_count: dict[int, int] = {}
    for st in geometry["streets"]:
        if not st["named"]:
            continue
        used += 1
        if used <= len(street_pool):
            name = street_pool[used - 1]
            aliases = sorted({name.replace("Road", "Rd"), name.replace("Street", "St"), name.lower(),
                              name.replace(" Road", " rd").replace(" Street", " st")} - {name})
        else:  # pool exhausted -> "3rd Cross, Gandhi Nagar" style (common in Indian towns)
            locs = [geometry["localities"][n] for n in st["nodes"] if n in geometry["localities"]]
            li = max(set(locs), key=locs.count)
            cross_count[li] = cross_count.get(li, 0) + 1
            c = cross_count[li]
            ordinal = f"{c}{'th' if 10 <= c % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(c % 10, 'th')}"
            loc_name = loc_pool[li % len(loc_pool)]
            name = f"{ordinal} Cross, {loc_name}"
            aliases = sorted({f"{ordinal} cross", f"{ordinal} cross {loc_name.lower()}", f"{c} cross {loc_name}"})
        pid = f"{prefix}-ST-{used:03d}"
        places.append({"place_id": pid, "name": name, "type": "street", "aliases": aliases,
                       "geometry": {"nodes": st["nodes"], "pipes": st["pipes"]}})
        for n in st["nodes"]:
            node_street.setdefault(n, pid)
    for li in range(geometry["n_localities"]):
        nodes = [n for n, l in geometry["localities"].items() if l == li]
        name = loc_pool[li % len(loc_pool)]
        places.append({"place_id": f"{prefix}-LC-{li:03d}", "name": name, "type": "locality",
                       "aliases": [name.lower()], "geometry": {"nodes": nodes}})
    for lm in geometry["landmarks"]:
        names = LANDMARK_NAMES[lm["type"]]
        name = names[(lm["name_idx"] + _stable_hash(town_id)) % len(names)] if town_id != "ltown" else names[lm["name_idx"]]
        aliases = sorted({a for a in lm["alias_words"]} | {name.lower()})
        places.append({"place_id": f"{prefix}-{lm['lid']}", "name": name, "type": "landmark", "subtype": lm["type"],
                       "aliases": aliases, "footfall": lm["footfall"],
                       "geometry": {"node": lm["node"]},
                       "locality": f"{prefix}-LC-{geometry['localities'][lm['node']]:03d}"})
    return {"town_id": town_id, "places": places}


def strip_geometry(gaz: dict) -> dict:
    """Version for the fine-tuning teammate: names, types, aliases only (docs/03 §6)."""
    out = []
    for p in gaz["places"]:
        q = {k: v for k, v in p.items() if k not in ("geometry", "footfall")}
        out.append(q)
    return {"town_id": gaz["town_id"], "places": out}


def save(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
