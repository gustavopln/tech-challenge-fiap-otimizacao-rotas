"""
Algoritmo Genético adaptado para otimização de rotas médicas
Baseado no código TSP original, com adição de restrições hospitalares
"""
import random
import math
import copy
from typing import List, Tuple
from models import Entrega, Veiculo, Base, Rota, PrioridadeEntrega


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calcula distância euclidiana entre dois pontos (pixels ou coordenadas)"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_route_metrics(sequencia: List[Tuple[float, float]], 
                           entregas: List[Entrega],
                           base: Base) -> Tuple[float, float, float]:
    """
    Calcula métricas de uma rota
    
    Returns:
        (distancia_total_pixels, distancia_total_km, tempo_total_min)
    """
    # Adiciona base no início e fim da sequência
    rota_completa = [base.localizacao] + sequencia + [base.localizacao]
    
    # Calcula distância em pixels
    distancia_pixels = 0
    for i in range(len(rota_completa) - 1):
        distancia_pixels += calculate_distance(rota_completa[i], rota_completa[i + 1])
    
    # Conversão aproximada: 1 pixel = 0.1 km (ajustar conforme necessário)
    PIXELS_TO_KM = 0.1
    distancia_km = distancia_pixels * PIXELS_TO_KM
    
    # Tempo total: viagem + paradas
    tempo_viagem = (distancia_km / 40) * 60  # 40 km/h em minutos
    tempo_paradas = sum(e.tempo_estimado_entrega_min for e in entregas)
    tempo_total = tempo_viagem + tempo_paradas
    
    return distancia_pixels, distancia_km, tempo_total


def calculate_fitness_with_constraints(sequencia: List[Tuple[float, float]],
                                      entregas: List[Entrega],
                                      veiculo: Veiculo,
                                      base: Base) -> float:
    """
    Calcula fitness considerando múltiplas restrições e objetivos
    
    Fitness = distância + penalidades
    Quanto MENOR, melhor (problema de minimização)
    """
    dist_pixels, dist_km, tempo_min = calculate_route_metrics(sequencia, entregas, base)
    
    fitness = dist_pixels  # Objetivo principal: minimizar distância
    
    # PENALIDADE 1: Violação de capacidade
    carga_total = sum(e.peso_kg for e in entregas)
    if carga_total > veiculo.capacidade_kg:
        excesso = carga_total - veiculo.capacidade_kg
        fitness += excesso * 1000  # Penalidade severa
    
    # PENALIDADE 2: Violação de autonomia
    if dist_km > veiculo.autonomia_km:
        excesso = dist_km - veiculo.autonomia_km
        fitness += excesso * 500
    
    # PENALIDADE 3: Prioridades não atendidas no início
    # Entregas críticas devem ser feitas primeiro
    peso_prioridade = 0
    for i, loc in enumerate(sequencia[:len(entregas)]):
        # Encontra a entrega correspondente
        entrega = next((e for e in entregas if e.localizacao == loc), None)
        if entrega:
            # Quanto mais crítica e mais tarde na rota, maior a penalidade
            if entrega.prioridade == PrioridadeEntrega.CRITICA:
                peso_prioridade += i * 100
            elif entrega.prioridade == PrioridadeEntrega.ALTA:
                peso_prioridade += i * 50
    
    fitness += peso_prioridade
    
    return fitness


def generate_random_population(entregas: List[Entrega], 
                              population_size: int) -> List[List[Tuple[float, float]]]:
    """
    Gera população inicial de rotas
    Mantém compatibilidade com código original (usa localizações)
    """
    localizacoes = [e.localizacao for e in entregas]
    return [random.sample(localizacoes, len(localizacoes)) for _ in range(population_size)]


def generate_priority_biased_population(entregas: List[Entrega], 
                                       population_size: int) -> List[List[Tuple[float, float]]]:
    """
    Gera população inicial com viés para prioridades
    50% da população começa com entregas críticas primeiro
    """
    population = []
    localizacoes = [e.localizacao for e in entregas]
    
    for i in range(population_size):
        if i < population_size // 2:
            # Metade com viés de prioridade
            sorted_entregas = sorted(entregas, key=lambda e: e.prioridade.value)
            sequencia = [e.localizacao for e in sorted_entregas]
            # Adiciona alguma aleatoriedade
            if len(sequencia) > 2:
                idx1 = random.randint(len(sequencia)//2, len(sequencia)-1)
                idx2 = random.randint(len(sequencia)//2, len(sequencia)-1)
                sequencia[idx1], sequencia[idx2] = sequencia[idx2], sequencia[idx1]
            population.append(sequencia)
        else:
            # Metade completamente aleatória
            population.append(random.sample(localizacoes, len(localizacoes)))
    
    return population


def order_crossover(parent1: List[Tuple[float, float]], 
                   parent2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Order Crossover (OX) - mantido do código original"""
    length = len(parent1)
    
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)
    
    child = parent1[start_index:end_index]
    
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]
    
    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)
    
    return child


def mutate(solution: List[Tuple[float, float]], 
          mutation_probability: float) -> List[Tuple[float, float]]:
    """
    Mutação por swap de posições adjacentes
    Mantido do código original
    """
    mutated_solution = copy.deepcopy(solution)
    
    if random.random() < mutation_probability:
        if len(solution) < 2:
            return solution
        
        index = random.randint(0, len(solution) - 2)
        mutated_solution[index], mutated_solution[index + 1] = \
            solution[index + 1], solution[index]
    
    return mutated_solution


def mutate_inversion(solution: List[Tuple[float, float]], 
                    mutation_probability: float) -> List[Tuple[float, float]]:
    """
    Mutação por inversão de segmento (mais agressiva)
    NOVA: útil para escapar de mínimos locais
    """
    mutated_solution = copy.deepcopy(solution)
    
    if random.random() < mutation_probability:
        if len(solution) < 3:
            return solution
        
        # Seleciona um segmento para inverter
        start = random.randint(0, len(solution) - 3)
        end = random.randint(start + 2, len(solution))
        
        # Inverte o segmento
        mutated_solution[start:end] = reversed(mutated_solution[start:end])
    
    return mutated_solution


def sort_population(population: List[List[Tuple[float, float]]], 
                   fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """Ordena população por fitness (mantido do original)"""
    combined = list(zip(population, fitness))
    sorted_combined = sorted(combined, key=lambda x: x[1])
    sorted_population, sorted_fitness = zip(*sorted_combined)
    return list(sorted_population), list(sorted_fitness)


# Exemplo de uso
if __name__ == '__main__':
    from models import Base, Veiculo, PrioridadeEntrega
    
    # Setup
    base = Base((400, 200), "Hospital Base")
    veiculo = Veiculo("V1", capacidade_kg=50.0, autonomia_km=100.0)
    
    # Entregas de exemplo
    entregas = [
        Entrega(1, (450, 250), "UBS 1", PrioridadeEntrega.CRITICA, 5.0),
        Entrega(2, (500, 150), "Clínica 2", PrioridadeEntrega.ALTA, 10.0),
        Entrega(3, (350, 300), "PSF 3", PrioridadeEntrega.MEDIA, 15.0),
        Entrega(4, (550, 200), "UPA 4", PrioridadeEntrega.BAIXA, 8.0),
    ]
    
    # Testa geração de população
    pop = generate_priority_biased_population(entregas, 10)
    print(f"População gerada: {len(pop)} indivíduos")
    
    # Testa fitness
    sequencia_teste = [e.localizacao for e in entregas]
    fitness = calculate_fitness_with_constraints(sequencia_teste, entregas, veiculo, base)
    print(f"\nFitness da sequência teste: {fitness:.2f}")
    
    # Testa mutação
    mutated = mutate_inversion(sequencia_teste, 1.0)
    print(f"\nOriginal:  {sequencia_teste}")
    print(f"Mutado:    {mutated}")
