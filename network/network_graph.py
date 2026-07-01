import os
import itertools
import networkx as nx
import matplotlib.pyplot as plt

from ner.person_ner import extract_persons


def create_network(comments):
    G = nx.Graph()

    for comment in comments:
        text = comment['comment']
        persons = extract_persons([text])
        
        names = [p['name'] for p in persons]
        names = list(set(names))

        if len(names) >= 2:
            for pair in itertools.combinations(names, 2):
                if G.has_edge(pair[0], pair[1]):
                    G[pair[0]][pair[1]]['weight'] += 1
                else:
                    G.add_edge(pair[0], pair[1], weight=1)

    # ---- PROSES PENGERJAAN VISUALISASI YANG LEBIH LAYAK ----
    
    # 1. Atur ukuran kanvas lebih besar (misal 14x10 inci) agar graf punya ruang bernapas
    plt.figure(figsize=(14, 10))

    # 2. Atur tata letak spring_layout dengan parameter k (jarak optimal antar node)
    # k yang semakin besar (default biasanya 1/sqrt(n)) akan memaksa node saling menjauh
    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)

    # 3. Ambil bobot (weight) untuk mengatur ketebalan garis (edges)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    # Normalisasi ketebalan garis agar tidak terlalu raksasa jika bobotnya puluhan
    max_weight = max(weights) if weights else 1
    edge_widths = [(w / max_weight) * 4 + 0.5 for w in weights] # ketebalan berkisar 0.5 - 4.5

    # 4. Gambar Node (Lingkaran) saja dengan ukuran yang jauh lebih kecil
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size=300,        # Diperkecil dari 3000 ke 300
        node_color='#2b7ce9', # Warna biru solid yang lebih modern
        alpha=0.8
    )

    # 5. Gambar Garis (Edges) dengan ketebalan dinamis berdasarkan bobot hubungan
    nx.draw_networkx_edges(
        G, 
        pos, 
        width=edge_widths, 
        edge_color='gray', 
        alpha=0.5
    )

    # 6. Gambar Teks Label dengan posisi agak digeser ke atas agar tidak menumpuk di dalam lingkaran
    # Menggunakan font_size lebih kecil dan background putih transparan (bbox) agar teks terbaca jelas
    label_pos = {k: [v[0], v[1] + 0.03] for k, v in pos.items()} # Geser posisi teks sedikit ke atas
    
    nx.draw_networkx_labels(
        G, 
        label_pos, 
        font_size=8, 
        font_family='sans-serif',
        font_weight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
    )

    plt.axis('off') # Menghilangkan sumbu x dan y kotak luar
    plt.tight_layout()

    # Pastikan folder static ada
    os.makedirs('static', exist_ok=True)
    
    plt.savefig('static/network.png', dpi=300, bbox_inches='tight') # Simpan dengan resolusi tinggi (dpi=300)
    plt.close()

    return 'network.png'