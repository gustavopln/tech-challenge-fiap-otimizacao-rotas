import pandas as pd
import random
import math

# -------------------------------------
# Configurações Iniciais
# -------------------------------------

# Coordenadas da base - Almoxarifado Central da Secretaria de Saúde do DF
BASE_COORD = (-15.8140447, -47.9659546)

# Arquivos de entrada
ARQ_VEICULOS = "data/veiculos.csv"
ARQ_HOSPITAIS = "data/hospitais_df.csv"

# Arquivos de saída
ARQ_ENTREGAS = "data/entregas.csv"
ARQ_ROTAS_INICIAIS = "data/rotas_iniciais.csv"

# Mapeamento de prioridades e penalidades
PRIORIDADES = {
    "CRITICA": 10.0,
    "ALTA": 5.0,
    "MEDIA": 2.0,
    "BAIXA": 1.0,
}


# -------------------------------------
# Funções auxiliares
# -------------------------------------

def calcular_distancia_km(p1, p2):
    """
    Calcula a distância aproximada em km entre dois pontos (lat, lon)
    usando a fórmula de Haversine.
    """
    R = 6371.0  # Raio médio da Terra em km

    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ------------------------------------------------------
# 1. Carregar veículos existentes (NÃO geramos mais)
# ------------------------------------------------------

try:
    df_veiculos = pd.read_csv(ARQ_VEICULOS)
except FileNotFoundError:
    raise SystemExit(f"Erro: arquivo '{ARQ_VEICULOS}' não encontrado.")

# Esperado: colunas -> id_veiculo, capacidade_kg, autonomia_km, velocidade_media_kmh, custo_por_km
if not {"id_veiculo", "capacidade_kg", "autonomia_km", "velocidade_media_kmh", "custo_por_km"}.issubset(df_veiculos.columns):
    raise SystemExit("Erro: o arquivo de veículos não possui as colunas esperadas.")


# ------------------------------------------------------------
# 2. Gerar Arquivo de Entregas a partir de hospitais_df.csv
# ------------------------------------------------------------

try:
    df_hospitais = pd.read_csv(ARQ_HOSPITAIS)
except FileNotFoundError:
    raise SystemExit(f"Erro: arquivo '{ARQ_HOSPITAIS}' não encontrado.")

# Esperado: colunas -> IdHospital, Nome, Latitude, Longitude
if not {"IdHospital", "Nome", "Latitude", "Longitude"}.issubset(df_hospitais.columns):
    raise SystemExit("Erro: o arquivo de hospitais não possui as colunas esperadas.")

entregas_data = []
id_entrega_seq = 1  # ID único da entrega (evento logístico)

for _, row in df_hospitais.iterrows():
    # Define prioridade (aqui aleatória; você pode sofisticar depois)
    prio_nome = random.choice(list(PRIORIDADES.keys()))

    entregas_data.append({
        "id": id_entrega_seq,                         # ID da entrega
        "id_hospital": int(row["IdHospital"]),        # ID do hospital
        "nome": row["Nome"],
        "lat": float(row["Latitude"]),
        "lng": float(row["Longitude"]),
        "prioridade": prio_nome,
        "peso_kg": round(random.uniform(10.0, 100.0), 2),
        "penalidade": PRIORIDADES[prio_nome],
        "tempo_estimado_entrega_min": random.choice([10, 15, 20, 30, 45, 60]),
    })

    id_entrega_seq += 1

df_entregas = pd.DataFrame(entregas_data)
df_entregas.to_csv(ARQ_ENTREGAS, index=False)
print(f"Arquivo de entregas gerado: {ARQ_ENTREGAS} (total: {len(df_entregas)} entregas)")


# -------------------------------------
# 3. Gerar Rotas Iniciais com base em veículos.csv + entregas.csv
# -------------------------------------

rotas_data = []
row_id = 1

num_veiculos = len(df_veiculos)
num_entregas = len(df_entregas)

if num_veiculos == 0 or num_entregas == 0:
    raise SystemExit("Erro: não há veículos ou entregas suficientes para gerar rotas.")

# Distribuir entregas aproximadamente de forma uniforme entre os veículos
entregas_por_veiculo = math.ceil(num_entregas / num_veiculos)

for i, v in df_veiculos.iterrows():
    inicio = i * entregas_por_veiculo
    fim = (i + 1) * entregas_por_veiculo
    subset_entregas = df_entregas.iloc[inicio:fim]

    if subset_entregas.empty:
        continue  # pode acontecer se sobrar veículo sem entrega em alguns cenários

    ponto_anterior = BASE_COORD

    for _, entrega in subset_entregas.iterrows():
        ponto_atual = (entrega["lat"], entrega["lng"])
        distancia = calcular_distancia_km(ponto_anterior, ponto_atual)

        # Custo = (Distância * Custo do Veículo) + Penalidade de Prioridade
        custo_segmento = (distancia * float(v["custo_por_km"])) + float(entrega["penalidade"])

        rotas_data.append({
            "id": row_id,
            "veiculo_id": v["id_veiculo"],
            "entrega_id": entrega["id"],
            "distancia_km": round(distancia, 3),
            "custo_segmento": round(custo_segmento, 2),
            "prioridade": entrega["prioridade"],
        })

        ponto_anterior = ponto_atual
        row_id += 1

df_rotas = pd.DataFrame(rotas_data)
df_rotas = df_rotas[["id", "veiculo_id", "entrega_id", "distancia_km", "custo_segmento", "prioridade"]]
df_rotas.to_csv(ARQ_ROTAS_INICIAIS, index=False)

print(f"Arquivo de rotas iniciais gerado: {ARQ_ROTAS_INICIAIS} (total: {len(df_rotas)} segmentos)")
