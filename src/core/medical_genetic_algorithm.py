"""
Algoritmo Genético para otimização de rotas médicas.

Este módulo implementa um Algoritmo Genético adaptado ao contexto
hospitalar, utilizando:

- Entrega, Veiculo, Base e Rota (modelos de domínio)
- Distância geográfica (Haversine) em km
- Penalidades por prioridade médica, capacidade e autonomia
- Seleção por torneio, crossover OX e mutação por inversão
- Elitismo e histórico de evolução do fitness

Integração principal:
- Leitura de veículos a partir de data/veiculos.csv
- Leitura de entregas a partir de data/entregas.csv (gerado pelo script gerar_csv_entregas_rotas_iniciais.py)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import pandas as pd

from src.models.models import Entrega, Veiculo, Base, Rota, PrioridadeEntrega
from src.models.base_default import BASE_PADRAO

# =====================================================================
# Utilitários de domínio e carregamento de dados
# =====================================================================

def distancia_haversine_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcula a distância aproximada em km entre dois pontos (lat, lon)
    usando a fórmula de Haversine.
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
    Carrega veículos a partir de um CSV no formato:
    id_veiculo, capacidade_kg, autonomia_km, velocidade_media_kmh, custo_por_km
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
    Carrega entregas a partir de data/entregas.csv, gerado pelo script
    gerar_csv_entregas_rotas_iniciais.py.

    Formato esperado (colunas mínimas):
        id, nome, lat, lng, prioridade, peso_kg, tempo_estimado_entrega_min

    Obs.: Se existir coluna 'penalidade', ela é ignorada aqui, pois a
    penalidade é derivada de PrioridadeEntrega.
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

        tempo_estimado = int(row["tempo_estimado_entrega_min"]) if "tempo_estimado_entrega_min" in df.columns else 15        

        entregas.append(
            Entrega(
                id_entrega=int(row["id"]),
                id_hospital=int(row["id_hospital"]),  # se quiser separar depois, pode vir de outra coluna
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
# Representação do indivíduo e métricas de rota
# =====================================================================


@dataclass
class Individual:
    """
    Representa um indivíduo na população do GA.
    O cromossomo é uma permutação dos índices das entregas.
    """

    ordem_entregas: List[int]  # índices na lista de entregas
    fitness: float | None = None


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

    # retorno à base (opcional, mas recomendado para cenário real)
    dist_back = distancia_haversine_km(ponto_anterior, base.localizacao)
    distancia_total_km += dist_back
    tempo_total_min += veiculo.tempo_viagem_minutos(dist_back)
    seq_coords.append(base.localizacao)

    return distancia_total_km, tempo_total_min, carga_total_kg, seq_coords


def calcular_fitness(
    individuo: Individual, entregas: List[Entrega], veiculo: Veiculo, base: Base
) -> float:
    """
    Função de fitness (minimização).

    Componentes:
    - custo operacional (distância_km * custo_por_km)
    - penalidade por criticidade das entregas
    - penalidade por excesso de carga
    - penalidade por violar autonomia do veículo
    """
    ordem = individuo.ordem_entregas
    distancia_km, tempo_min, carga_total_kg, _ = calcular_metricas_rota(
        ordem, entregas, base, veiculo
    )

    # Custo operacional base
    custo_operacional = distancia_km * veiculo.custo_por_km

    # Penalidade por prioridade médica (todas as entregas da rota)
    penalidade_prioridade = sum(
        entregas[i].prioridade.peso_penalidade()
        for i in ordem
    )

    fitness = custo_operacional + penalidade_prioridade

    # Penalidade por excesso de carga
    if carga_total_kg > veiculo.capacidade_kg:
        excesso = carga_total_kg - veiculo.capacidade_kg
        fitness += excesso * 1000.0  # penalidade forte

    # Penalidade por violar autonomia
    if distancia_km > veiculo.autonomia_km:
        excesso = distancia_km - veiculo.autonomia_km
        fitness += excesso * 500.0  # penalidade forte

    # (Opcional futuro) penalidade por janelas de tempo

    individuo.fitness = fitness
    return fitness


# =====================================================================
# Operadores Genéticos: população, seleção, crossover, mutação
# =====================================================================


def gerar_populacao_inicial(
    num_individuos: int, num_entregas: int
) -> List[Individual]:
    """
    Gera uma população inicial de permutações.

    Simples: permutações aleatórias.
    Pode ser refinado com viés por prioridade depois.
    """
    base_indices = list(range(num_entregas))
    populacao: List[Individual] = []

    for _ in range(num_individuos):
        ordem = base_indices[:]
        random.shuffle(ordem)
        populacao.append(Individual(ordem_entregas=ordem))

    return populacao


def selecao_torneio(
    populacao: List[Individual], k: int = 3
) -> Individual:
    """
    Seleção por torneio: escolhe k indivíduos aleatórios e retorna o melhor.
    """
    escolhidos = random.sample(populacao, k)
    escolhido = min(escolhidos, key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"))
    return escolhido


def order_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Order Crossover (OX) clássico para permutações.

    - Seleciona um segmento do pai 1 e o copia para o filho
    - Preenche o restante na ordem em que aparecem no pai 2
    """
    p1 = parent1.ordem_entregas
    p2 = parent2.ordem_entregas
    n = len(p1)

    if n < 2:
        return Individual(p1[:]), Individual(p2[:])

    i, j = sorted(random.sample(range(n), 2))
    # Segmento do pai 1
    segmento = p1[i:j]

    def construir_filho(seg: List[int], p_seg: Sequence[int]) -> List[int]:
        filho = [None] * n  # type: ignore
        # Copia segmento
        filho[i:j] = seg
        pos = j
        for gene in p_seg:
            if gene in seg:
                continue
            if pos >= n:
                pos = 0
            filho[pos] = gene
            pos += 1
        # type: ignore
        return filho  # type: ignore

    filho1_ordem = construir_filho(segmento, p2)
    filho2_ordem = construir_filho(segmento, p1)

    return Individual(filho1_ordem), Individual(filho2_ordem)


def mutacao_inversao(individuo: Individual, taxa_mutacao: float) -> None:
    """
    Mutação por inversão: escolhe um segmento e inverte a ordem dos genes.
    """
    if random.random() > taxa_mutacao:
        return

    ordem = individuo.ordem_entregas
    n = len(ordem)
    if n < 2:
        return

    i, j = sorted(random.sample(range(n), 2))
    ordem[i:j] = reversed(ordem[i:j])


# =====================================================================
# Execução do Algoritmo Genético
# =====================================================================


@dataclass
class GAConfig:
    tamanho_populacao: int = 100
    geracoes: int = 200
    taxa_mutacao: float = 0.1
    elitismo: float = 0.1  # proporção de indivíduos mantidos (0.0–0.5)


def executar_ga_para_veiculo(
    entregas: List[Entrega],
    veiculo: Veiculo,
    base: Base,
    config: GAConfig | None = None,
) -> Tuple[Rota, List[float]]:
    """
    Executa o GA para otimizar a rota de UM veículo atendendo TODAS as entregas.

    Retorna:
    - Rota com melhor solução encontrada
    - lista com o melhor fitness de cada geração (para gráficos/análise)
    """
    if config is None:
        config = GAConfig()

    num_entregas = len(entregas)
    if num_entregas == 0:
        raise ValueError("Não há entregas para otimizar.")

    # População inicial
    populacao = gerar_populacao_inicial(config.tamanho_populacao, num_entregas)

    melhor_fitness_por_geracao: List[float] = []
    melhor_individuo_global: Individual | None = None

    # Avaliação inicial
    for ind in populacao:
        calcular_fitness(ind, entregas, veiculo, base)

    for gen in range(config.geracoes):
        # Ordena por fitness (menor é melhor)
        populacao.sort(key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"))

        melhor_geracao = populacao[0]
        melhor_fitness_por_geracao.append(melhor_geracao.fitness or float("inf"))

        if melhor_individuo_global is None or (melhor_geracao.fitness or float("inf")) < (melhor_individuo_global.fitness or float("inf")):
            melhor_individuo_global = Individual(
                ordem_entregas=melhor_geracao.ordem_entregas[:],
                fitness=melhor_geracao.fitness,
            )

        # Elitismo
        num_elite = max(1, int(config.elitismo * config.tamanho_populacao))
        novos_individuos: List[Individual] = [
            Individual(ind.ordem_entregas[:], ind.fitness)
            for ind in populacao[:num_elite]
        ]

        # Geração de novos indivíduos
        while len(novos_individuos) < config.tamanho_populacao:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho1, filho2 = order_crossover(pai1, pai2)

            mutacao_inversao(filho1, config.taxa_mutacao)
            mutacao_inversao(filho2, config.taxa_mutacao)

            calcular_fitness(filho1, entregas, veiculo, base)
            calcular_fitness(filho2, entregas, veiculo, base)

            novos_individuos.append(filho1)
            if len(novos_individuos) < config.tamanho_populacao:
                novos_individuos.append(filho2)

        populacao = novos_individuos

    # Melhor indivíduo global
    assert melhor_individuo_global is not None
    melhor_ordem = melhor_individuo_global.ordem_entregas

    distancia_km, tempo_min, carga_total_kg, seq_coords = calcular_metricas_rota(
        melhor_ordem, entregas, base, veiculo
    )

    melhor_entregas = [entregas[i] for i in melhor_ordem]

    rota = Rota(
        veiculo=veiculo,
        entregas=melhor_entregas,
        sequencia=seq_coords,
        distancia_total_km=distancia_km,
        tempo_total_min=tempo_min,
    )

    return rota, melhor_fitness_por_geracao


# =====================================================================
# Bloco de teste rápido (opcional, para desenvolvimento)
# =====================================================================

if __name__ == "__main__":
    """
    Execução rápida de teste do GA usando:

    - data/veiculos.csv
    - data/entregas.csv

    Este bloco não deve ser usado em produção, apenas para inspeção manual.
    """
    base = BASE_PADRAO
    entregas = carregar_entregas_csv("data/entregas.csv")
    veiculos = carregar_veiculos_csv("data/veiculos.csv")

    veiculo = veiculos[0]  # testa com o primeiro veículo

    config = GAConfig(
        tamanho_populacao=50,
        geracoes=50,
        taxa_mutacao=0.2,
        elitismo=0.1,
    )

    melhor_rota, historico = executar_ga_para_veiculo(
        entregas=entregas,
        veiculo=veiculo,
        base=base,
        config=config,
    )

    print(f"Melhor rota para o veículo {veiculo.id_veiculo}:")
    print(melhor_rota)
    print(f"Melhor fitness final: {historico[-1]:.2f}")
    print(f"Distância total: {melhor_rota.distancia_total_km:.2f} km")
    print(f"Carga total: {melhor_rota.carga_total_kg:.2f} kg")
