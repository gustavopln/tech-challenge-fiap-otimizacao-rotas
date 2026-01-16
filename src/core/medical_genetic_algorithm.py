"""
Algoritmo Genético para otimização de rotas médicas.

Este módulo implementa um Algoritmo Genético adaptado ao contexto hospitalar, utilizando:

- Entrega, Veiculo, Base e Rota (modelos de domínio)
- Distância geográfica (Haversine) em km
- Penalidades por prioridade médica, capacidade e autonomia
- Seleção, crossover OX e mutações sobre a representação VRP
- Elitismo e histórico de evolução do fitness

Integração principal:
- Leitura de veículos a partir de data/veiculos.csv
- Leitura de entregas a partir de data/entregas.csv (gerado pelo script gerar_csv_entregas_rotas_iniciais.py)
- Uso de rotas_iniciais.csv como semente para população inicial do GA (população semeada)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict

import pandas as pd

from src.visualization.gerar_mapa_rotas_vrp import gerar_mapa_rotas_vrp
from src.models.models import Entrega, Veiculo, Base, Rota, PrioridadeEntrega
from src.models.base_default import BASE_PADRAO
from src.utils.relatorios_ga import (
    GAEvolutionReport,
    relatorio_populacao_inicial_vs_final,
    exportar_historico_para_csv
)

# =====================================================================
# Utilitários de domínio e carregamento de dados
# =====================================================================

def distancia_haversine_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcula a distância aproximada em km entre dois pontos (lat, lon) usando a fórmula de Haversine.
    """
    R = 6371.0  # raio médio da Terra em km

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


def carregar_veiculos_csv(path: str) -> List[Veiculo]:
    """
    Carrega veículos a partir de um CSV no formato: id_veiculo, capacidade_kg, autonomia_km, velocidade_media_kmh, custo_por_km
    """
    df = pd.read_csv(path)

    esperadas = {
        "id_veiculo",
        "capacidade_kg",
        "autonomia_km",
        "velocidade_media_kmh",
        "custo_por_km",
    }

    if not esperadas.issubset(df.columns):
        raise ValueError(
            f"Arquivo de veículos {path} não possui colunas esperadas: {esperadas}"
        )

    veiculos: List[Veiculo] = []
    for _, row in df.iterrows():
        veiculos.append(
            Veiculo(
                id_veiculo=str(row["id_veiculo"]),
                capacidade_kg=float(row["capacidade_kg"]),
                autonomia_km=float(row["autonomia_km"]),
                velocidade_media_kmh=float(row["velocidade_media_kmh"]),
                custo_por_km=float(row["custo_por_km"]),
            )
        )
    return veiculos


def carregar_entregas_csv(path: str) -> List[Entrega]:
    """
    Carrega entregas a partir de data/entregas.csv, gerado pelo script gerar_csv_entregas_rotas_iniciais.py.
    Formato esperado (colunas mínimas): id, id_hospital, nome, lat, lng, prioridade, peso_kg, tempo_estimado_entrega_min
    Obs.: Se existir coluna 'penalidade', ela é ignorada aqui, pois a penalidade é derivada de PrioridadeEntrega.
    """
    df = pd.read_csv(path)

    esperadas_minimas = {
        "id",
        "id_hospital",
        "nome",
        "lat",
        "lng",
        "prioridade",
        "peso_kg",
        "tempo_estimado_entrega_min",
    }

    if not esperadas_minimas.issubset(df.columns):
        raise ValueError(
            f"Arquivo de entregas {path} não possui colunas esperadas mínimas: {esperadas_minimas}"
        )
    
    entregas: List[Entrega] = []
    
    for _, row in df.iterrows():
        # Converte string ("CRITICA", "ALTA", etc.) para enum PrioridadeEntrega
        prioridade_str = str(row["prioridade"]).upper().strip()
        prioridade = PrioridadeEntrega[prioridade_str]

        tempo_estimado = int(row["tempo_estimado_entrega_min"])

        entregas.append(
            Entrega(
                id_entrega=int(row["id"]),
                id_hospital=int(row["id_hospital"]),
                nome=str(row["nome"]),
                localizacao=(float(row["lat"]), float(row["lng"])),
                prioridade=prioridade,
                peso_kg=float(row["peso_kg"]),
                tempo_estimado_entrega_min=tempo_estimado,
                janela_inicio=0,
                janela_fim=1440,
            )
        )

    return entregas


# =====================================================================
# Métricas de rota (utilizadas pelo VRP)
# =====================================================================

def calcular_metricas_rota(
    ordem: Sequence[int], entregas: List[Entrega], base: Base, veiculo: Veiculo
) -> Tuple[float, float, float, List[Tuple[float, float]]]:
    """
    Calcula:
    - distância total em km
    - tempo total em minutos (viagem + paradas)
    - carga total em kg
    - sequência de coordenadas visitadas

    Considera:
    - saída da base -> primeira entrega -> ... -> última entrega -> retorno à base
    """
    if not ordem:
        return 0.0, 0.0, 0.0, [base.localizacao]

    distancia_total_km = 0.0
    tempo_total_min = 0.0
    carga_total_kg = 0.0

    seq_coords: List[Tuple[float, float]] = [base.localizacao]
    ponto_anterior = base.localizacao

    for idx in ordem:
        entrega = entregas[idx]
        ponto_atual = entrega.localizacao

        dist_km = distancia_haversine_km(ponto_anterior, ponto_atual)
        distancia_total_km += dist_km
        tempo_total_min += veiculo.tempo_viagem_minutos(dist_km)
        tempo_total_min += entrega.tempo_estimado_entrega_min
        carga_total_kg += entrega.peso_kg

        seq_coords.append(ponto_atual)
        ponto_anterior = ponto_atual

    # retorno à base
    dist_back = distancia_haversine_km(ponto_anterior, base.localizacao)
    distancia_total_km += dist_back
    tempo_total_min += veiculo.tempo_viagem_minutos(dist_back)
    seq_coords.append(base.localizacao)

    return distancia_total_km, tempo_total_min, carga_total_kg, seq_coords

# =====================================================================
# Representação VRP: múltiplos veículos
# =====================================================================

@dataclass
class IndividualVRP:
    """
    Indivíduo para VRP (múltiplos veículos).

    - ordem_entregas: permutação de índices de entregas (0..N-1)
    - cortes: lista de posições onde a sequência é cortada para separar as rotas dos veículos.

    Exemplo:
      ordem_entregas = [3, 1, 9, 4, 2, 0, 8, 7, 6, 5]
      cortes = [3, 7]

      => Veículo 1: [3, 1, 9]
      => Veículo 2: [4, 2, 0, 8]
      => Veículo 3: [7, 6, 5]
    """

    ordem_entregas: List[int]
    cortes: List[int]
    fitness: float | None = None

def decodificar_individuo_vrp(
    individuo: IndividualVRP,
    num_veiculos: int,
) -> List[List[int]]:
    """
    Converte (ordem_entregas + cortes) em uma lista de rotas por veículo.

    Retorna:
        lista de listas de índices de entregas, uma por veículo.
    """
    ordem = individuo.ordem_entregas
    cortes = sorted(c for c in individuo.cortes if 0 < c < len(ordem))

    rotas: List[List[int]] = []
    inicio = 0

    for c in cortes:
        rotas.append(ordem[inicio:c])
        inicio = c

    rotas.append(ordem[inicio:])

    # Ajusta quantidade de rotas para bater com num_veiculos
    if len(rotas) < num_veiculos:
        # completa com veículos sem entregas
        rotas.extend([[] for _ in range(num_veiculos - len(rotas))])
    elif len(rotas) > num_veiculos:
        # se tiver mais segmentos que veículos, cola os extras no último
        extras: List[int] = []
        for r in rotas[num_veiculos - 1:]:
            extras.extend(r)
        rotas = rotas[: num_veiculos - 1] + [rotas[num_veiculos - 1] + extras]

    return rotas

def construir_individuo_vrp_de_rotas_iniciais(
    entregas: List[Entrega],
    veiculos: List[Veiculo],
    caminho_rotas: str,
) -> IndividualVRP:
    """
    Constrói um indivíduo VRP a partir de data/rotas_iniciais.csv.

    O arquivo rotas_iniciais.csv é gerado pelo script
    src/utils/gerar_csv_entregas_rotas_iniciais.py e contém, no mínimo:
        id, veiculo_id, entrega_id, distancia_km, custo_segmento, prioridade

    Estratégia:
    - Agrupa as entregas por veiculo_id na ordem do CSV.
    - Para cada veículo em 'veiculos', recupera a lista de entregas correspondente.
    - Concatena todas as listas em uma única permutação global.
    - Define cortes no final de cada rota de veículo.
    """
    df = pd.read_csv(caminho_rotas)

    # Mapa id_entrega -> índice na lista 'entregas'
    mapa_id_para_indice: Dict[int, int] = {
        e.id_entrega: idx for idx, e in enumerate(entregas)
    }

    # Agrupa entrega_id por veiculo_id na ordem do arquivo
    rotas_por_veiculo_id: Dict[str, List[int]] = {}
    df_ordenado = df.sort_values(["veiculo_id", "id"])

    for _, row in df_ordenado.iterrows():
        vid = str(row["veiculo_id"]).strip()
        entrega_id = int(row["entrega_id"])

        if entrega_id not in mapa_id_para_indice:
            # Se por algum motivo houver entrega_id não presente em 'entregas',
            # simplesmente ignora (proteção básica).
            continue

        idx_entrega = mapa_id_para_indice[entrega_id]

        if vid not in rotas_por_veiculo_id:
            rotas_por_veiculo_id[vid] = []
        rotas_por_veiculo_id[vid].append(idx_entrega)

    ordem_global: List[int] = []
    cortes: List[int] = []
    acumulado = 0

    # Respeita a ordem dos veículos carregados em 'veiculos'
    for veiculo in veiculos:
        vid = str(veiculo.id_veiculo).strip()
        lista_indices = rotas_por_veiculo_id.get(vid, [])

        if not lista_indices:
            # veículo sem rota inicial explícita
            continue

        ordem_global.extend(lista_indices)
        acumulado += len(lista_indices)
        cortes.append(acumulado)

    # Remove o último corte (não precisamos de corte "após a última rota")
    if cortes:
        cortes = cortes[:-1]

    if not ordem_global:
        raise ValueError(
            f"Não foi possível construir indivíduo VRP a partir de {caminho_rotas}. "
            "Verifique se veiculo_id e entrega_id batem com os CSVs de veículos e entregas."
        )

    return IndividualVRP(ordem_entregas=ordem_global, cortes=cortes)

def gerar_populacao_inicial_vrp(
    tamanho_populacao: int,
    entregas: List[Entrega],
    veiculos: List[Veiculo],
    caminho_rotas_iniciais: str,
) -> List[IndividualVRP]:
    """
    Gera população inicial para VRP 100% SEMEADA a partir de rotas_iniciais.csv.

    - Constrói um indivíduo base (semente) a partir das rotas existentes.
    - A população inicial é composta apenas por cópias desse indíviduo.

    A diversidade será introduzida ao longo das gerações por meio de crossover e mutações, sempre partindo dessa configuração semada.
    """
    # Tenta construir indivíduo a partir das rotas iniciais    
    individuo_base = construir_individuo_vrp_de_rotas_iniciais(
        entregas=entregas,
        veiculos=veiculos,
        caminho_rotas=caminho_rotas_iniciais,
    )

    if tamanho_populacao < 1:
        return [individuo_base]

    populacao: List[IndividualVRP] = [individuo_base]

    # Preenche a população com cópias da semente
    for _ in range(tamanho_populacao - 1):
        populacao.append(
            IndividualVRP(
                ordem_entregas=individuo_base.ordem_entregas[:],
                cortes=individuo_base.cortes[:],
                fitness=None,
            )
        )

    return populacao

def calcular_fitness_vrp(
    individuo: IndividualVRP,
    entregas: List[Entrega],
    veiculos: List[Veiculo],
    base: Base,
) -> float:
    """
    Função de fitness para VRP (múltiplos veículos).

    Soma:
    - custo operacional de cada veículo (distância * custo_por_km)
    - penalidade de prioridade das entregas
    - penalidade por excesso de carga
    - penalidade por violar autonomia

    Quanto MENOR o fitness, melhor a solução.
    """
    rotas_indices = decodificar_individuo_vrp(individuo, num_veiculos=len(veiculos))

    fitness_total = 0.0

    for rota_idx, veiculo in zip(rotas_indices, veiculos):
        if not rota_idx:
            # veículo sem entregas: custo zero
            continue

        distancia_km, tempo_min, carga_total_kg, _ = calcular_metricas_rota(
            rota_idx, entregas, base, veiculo
        )

        # custo base (distância * custo do veículo)
        custo_operacional = distancia_km * veiculo.custo_por_km

        # penalidade de prioridade (somatório sobre as entregas daquela rota)
        penalidade_prioridade = sum(
            entregas[i].prioridade.peso_penalidade() for i in rota_idx
        )

        fitness = custo_operacional + penalidade_prioridade

        # penalidade por excesso de carga
        if carga_total_kg > veiculo.capacidade_kg:
            excesso = carga_total_kg - veiculo.capacidade_kg
            fitness += excesso * 1000.0

        # penalidade por violar autonomia
        if distancia_km > veiculo.autonomia_km:
            excesso = distancia_km - veiculo.autonomia_km
            fitness += excesso * 500.0

        fitness_total += fitness

    individuo.fitness = fitness_total
    return fitness_total

# =====================================================================
# Operadores genéticos VRP
# =====================================================================

def order_crossover_ordens(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Order Crossover (OX) aplicado diretamente em listas de inteiros.
    Retorna duas novas listas (filhos).
    """
    n = len(p1)
    if n < 2:
        return p1[:], p2[:]

    i, j = sorted(random.sample(range(n), 2))
    segmento = p1[i:j]

    def construir_filho(seg: List[int], outro: List[int]) -> List[int]:
        filho: List[int | None] = [None] * n
        filho[i:j] = seg
        pos = j
        for gene in outro:
            if gene in seg:
                continue
            while filho[pos] is not None:
                pos = (pos + 1) % n
            filho[pos] = gene
            pos = (pos + 1) % n
        return [g for g in filho if g is not None]  # type: ignore

    f1 = construir_filho(segmento, p2)
    f2 = construir_filho(segmento, p1)

    return f1, f2

def crossover_vrp(
    parent1: IndividualVRP,
    parent2: IndividualVRP,
) -> Tuple[IndividualVRP, IndividualVRP]:
    """
    Crossover para VRP:
    - OX na permutação de entregas
    - cortes herdados aleatoriamente de um dos pais
    """
    f1_ordem, f2_ordem = order_crossover_ordens(
        parent1.ordem_entregas, parent2.ordem_entregas
    )

    # Herdar cortes de um dos pais (simples e eficaz)
    f1_cortes = parent1.cortes[:] if random.random() < 0.5 else parent2.cortes[:]
    f2_cortes = parent1.cortes[:] if random.random() < 0.5 else parent2.cortes[:]

    return (
        IndividualVRP(ordem_entregas=f1_ordem, cortes=f1_cortes),
        IndividualVRP(ordem_entregas=f2_ordem, cortes=f2_cortes),
    )

def mutacao_inversao_vrp(individuo: IndividualVRP, taxa_mutacao: float) -> None:
    """
    Mutação de inversão na permutação de entregas.
    """
    if random.random() > taxa_mutacao:
        return

    ordem = individuo.ordem_entregas
    n = len(ordem)
    if n < 2:
        return

    i, j = sorted(random.sample(range(n), 2))
    ordem[i:j] = reversed(ordem[i:j])

def mutacao_cortes(
    individuo: IndividualVRP,
    num_entregas: int,
    taxa_mutacao: float,
) -> None:
    """
    Mutação simples nos cortes:
    - escolhe um corte e desloca uma posição para esquerda ou direita.
    """
    if random.random() > taxa_mutacao:
        return

    if not individuo.cortes:
        return

    idx = random.randrange(len(individuo.cortes))
    delta = random.choice([-1, 1])
    novo = individuo.cortes[idx] + delta
    novo = max(1, min(num_entregas - 1, novo))

    individuo.cortes[idx] = novo
    individuo.cortes = sorted(set(individuo.cortes))


# =====================================================================
# Execução do Algoritmo Genético
# =====================================================================

@dataclass
class GAConfig:
    tamanho_populacao: int = 100
    geracoes: int = 200
    taxa_mutacao: float = 0.1
    elitismo: float = 0.1  # proporção de indivíduos mantidos (0.0–0.5)


def executar_ga_vrp(
    entregas: List[Entrega],
    veiculos: List[Veiculo],
    base: Base,
    config: GAConfig | None = None,    
    caminho_rotas_iniciais: str = "data/rotas_iniciais.csv",
) -> Tuple[List[Rota], List[float]]:
    """
    Executa o AG para VRP (múltiplos veículos).

    Retorna:
    - lista de Rotas (uma por veículo)
    - histórico do melhor fitness por geração
    """
    if config is None:
        config = GAConfig()

    num_entregas = len(entregas)
    num_veiculos = len(veiculos)

    if num_entregas == 0:
        raise ValueError("Não há entregas para otimizar.")
    if num_veiculos == 0:
        raise ValueError("Não há veículos disponíveis.")

    
    populacao = gerar_populacao_inicial_vrp(
        tamanho_populacao=config.tamanho_populacao,
        entregas=entregas,
        veiculos=veiculos,
        caminho_rotas_iniciais=caminho_rotas_iniciais,
    )
    

    melhor_fitness_por_geracao: List[float] = []
    melhor_individual_global: IndividualVRP | None = None

    # Avaliação inicial
    for ind in populacao:
        calcular_fitness_vrp(ind, entregas, veiculos, base)

    for _ in range(config.geracoes):
        populacao.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float("inf")
        )

        melhor_geracao = populacao[0]
        melhor_fitness_por_geracao.append(melhor_geracao.fitness or float("inf"))

        if (
            melhor_individual_global is None
            or (melhor_geracao.fitness or float("inf")) < (melhor_individual_global.fitness or float("inf"))
        ):
            melhor_individual_global = IndividualVRP(
                ordem_entregas=melhor_geracao.ordem_entregas[:],
                cortes=melhor_geracao.cortes[:],
                fitness=melhor_geracao.fitness,
            )

        # Elitismo
        num_elite = max(1, int(config.elitismo * config.tamanho_populacao))
        novos: List[IndividualVRP] = [
            IndividualVRP(
                ordem_entregas=ind.ordem_entregas[:],
                cortes=ind.cortes[:],
                fitness=ind.fitness,
            )
            for ind in populacao[:num_elite]
        ]

        # Geração de novos indivíduos
        while len(novos) < config.tamanho_populacao:
            pai1 = random.choice(populacao)
            pai2 = random.choice(populacao)
            filho1, filho2 = crossover_vrp(pai1, pai2)

            mutacao_inversao_vrp(filho1, config.taxa_mutacao)
            mutacao_cortes(filho1, num_entregas, config.taxa_mutacao)
            calcular_fitness_vrp(filho1, entregas, veiculos, base)
            novos.append(filho1)

            if len(novos) < config.tamanho_populacao:
                mutacao_inversao_vrp(filho2, config.taxa_mutacao)
                mutacao_cortes(filho2, num_entregas, config.taxa_mutacao)
                calcular_fitness_vrp(filho2, entregas, veiculos, base)
                novos.append(filho2)

        populacao = novos

    # Melhor indivíduo global
    assert melhor_individual_global is not None
    melhor_rotas_indices = decodificar_individuo_vrp(
        melhor_individual_global, num_veiculos=num_veiculos
    )

    rotas_resultado: List[Rota] = []

    for rota_idx, veiculo in zip(melhor_rotas_indices, veiculos):
        if not rota_idx:
            # veículo sem rota
            rotas_resultado.append(
                Rota(
                    veiculo=veiculo,
                    entregas=[],
                    sequencia=[base.localizacao],
                    distancia_total_km=0.0,
                    tempo_total_min=0.0,
                )
            )
            continue

        distancia_km, tempo_min, _, seq_coords = calcular_metricas_rota(
            rota_idx, entregas, base, veiculo
        )
        entregas_rota = [entregas[i] for i in rota_idx]

        rotas_resultado.append(
            Rota(
                veiculo=veiculo,
                entregas=entregas_rota,
                sequencia=seq_coords,
                distancia_total_km=distancia_km,
                tempo_total_min=tempo_min,
            )
        )

    return rotas_resultado, melhor_fitness_por_geracao


# =====================================================================
# Bloco de teste rápido
# =====================================================================

if __name__ == "__main__":
    """
    Execução rápida de teste do GA usando:

    - data/veiculos.csv
    - data/entregas.csv
    - data/rotas_iniciais.csv (semente da população)

    Este bloco não deve ser usado em produção, apenas para inspeção manual e geração de material de análise/relatório.
    """
    base = BASE_PADRAO
    entregas = carregar_entregas_csv("data/entregas.csv")
    veiculos = carregar_veiculos_csv("data/veiculos.csv")

    # Exemplo: usar todos os veículos do CSV
    config = GAConfig(
        tamanho_populacao=150, # mantêm diversidade suficiente
        geracoes=1000, # garante tempo para explorar várias combinações
        taxa_mutacao=0.25, # evita que o GA fique preso em mínimos locais
        elitismo=0.03, # preserva bons indivíduos, mas sem congelar o GA
    )

    print("=" * 56)
    print("=== VRP com população semeada por rotas_iniciais.csv ===")
    print("=" * 56)
    rotas_semeado, historico_semeado = executar_ga_vrp(
        entregas=entregas,
        veiculos=veiculos,
        base=base,
        config=config,        
        caminho_rotas_iniciais="data/rotas_iniciais.csv",
    )
    print(f"Melhor fitness final (VRP - semeado): {historico_semeado[-1]:.2f}\n")

    print("=" * 44)
    print("=== Detalhe das rotas no cenário semeado ===")
    print("=" * 44)
    for rota in rotas_semeado:
        print(rota)
        print(f"  Distância total: {rota.distancia_total_km:.2f} km")
        print(f"  Carga total: {rota.carga_total_kg:.2f} kg")
        print(f"  Tempo total: {rota.tempo_total_min:.2f} min\n")
        print()

    # -----------------------------------------
    # Relatórios de evolução do GA (cenário semeado)
    # -----------------------------------------
    report = GAEvolutionReport(
        nome_cenario="VRP - população semeada (rotas_iniciais.csv)",
        historico_fitness=historico_semeado,
    )

    # print()
    # print(report.resumo_texto())
    # print()
    print(relatorio_populacao_inicial_vs_final(
        "VRP - população semeada (rotas_iniciais.csv)",
        historico_semeado,
    ))

    # Exporta histórico para CSV para análise em notebook / relatório
    exportar_historico_para_csv(
        "data/resultados/historico_ga_vrp_semeado.csv",
        historico_semeado,
        nome_coluna="fitness_melhor_individuo",
    )
    print("\nHistórico de fitness exportado para data/resultados/historico_ga_vrp_semeado.csv")
    
    # Geração automática do mapa VRP a partir das rotas semeadas
    gerar_mapa_rotas_vrp(    
        rotas_semeado,
        nome_arquivo="data/resultados/mapa_rotas_vrp.html",
    )    