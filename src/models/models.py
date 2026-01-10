"""
Modelos de dados para o sistema de otimização de rotas médicas
"""
from dataclasses import dataclass, field
from typing import Tuple, List
from enum import Enum

class PrioridadeEntrega(Enum):
    """
    Níveis de prioridade para entregas médicas.

    A enumeração em si representa apenas o nível lógico de prioridade.
    O peso utilizado na função de custo/fitness é obtido via
    o método `peso_penalidade`.
    """
    CRITICA = 1      # Medicamentos críticos, emergências
    ALTA = 2         # Medicamentos controlados, tempo-sensíveis
    MEDIA = 3        # Insumos importantes
    BAIXA = 4        # Insumos regulares

    def peso_penalidade(self) -> float:
        """
        Peso usado na função de custo/fitness.
        Entregas mais críticas recebem maior penalidade, garantindo que o algoritmo genético priorize naturalmente essas entregas.
        """
        return {
            PrioridadeEntrega.CRITICA: 10.0,
            PrioridadeEntrega.ALTA: 5.0,
            PrioridadeEntrega.MEDIA: 2.0,
            PrioridadeEntrega.BAIXA: 1.0,
        }[self]
    

@dataclass(frozen=True)
class Entrega:
    """
    Representa um evento de entrega em um ponto de atendimento (hospital/UBS/UPA).

    Importante:
    - `id_entrega` é um identificador único da entrega (evento logístico).
    - `id_hospital` identifica o local de entrega (pode haver várias entregas
      para o mesmo hospital ao longo do tempo).
    """

    id_entrega: int
    id_hospital: int
    nome: str
    localizacao: Tuple[float, float]  # (latitude, longitude)
    prioridade: PrioridadeEntrega
    peso_kg: float
    tempo_estimado_entrega_min: int = 10  # Tempo médio de parada
    janela_inicio: int = 0  # Hora início (em minutos desde 00:00)
    janela_fim: int = 1440  # Hora fim (padrão: 24h = 1440 min)
    
    def __repr__(self) -> str:
        return (
            f"Entrega(id={self.id_entrega}, "
            f"hospital={self.id_hospital}, "
            f"nome='{self.nome}', "
            f"prioridade={self.prioridade.name}, "
            f"peso={self.peso_kg}kg)"
        )

@dataclass(frozen=True)
class Veiculo:
    """
    Representa um veículo de entrega.

    - `capacidade_kg`: capacidade máxima de carga.
    - `autonomia_km`: autonomia máxima (ida + volta) sem reabastecer.
    - `velocidade_media_kmh`: usada para cálculo aproximado de tempo.
    - `custo_por_km`: custo operacional estimado por quilômetro.
    """

    id_veiculo: str
    capacidade_kg: float
    autonomia_km: float
    velocidade_media_kmh: float = 40.0
    custo_por_km: float = 2.50
    
    def tempo_viagem_minutos(self, distancia_km: float) -> float:
        """
        Calcula o tempo estimado de viagem em minutos para uma distância em km.
        """
        if self.velocidade_media_kmh <= 0:
            return 0.0
        return (distancia_km / self.velocidade_media_kmh) * 60.0
    
    def pode_carregar(self, peso_kg: float) -> bool:
        """
        Verifica se o veículo pode carregar determinado peso.
        """
        return peso_kg <= self.capacidade_kg
    
    def validar_carga(self, peso_kg: float):
        """
        Retorna (ok, mensagem) para verificação de carga.
        """
        if peso_kg > self.capacidade_kg:
            excesso = peso_kg - self.capacidade_kg
            return False, f"Excede capacidade do veículo em {excesso:.2f} kg"
        return True, "OK"
    
    def pode_percorrer(self, distancia_km: float) -> bool:
        """
        Verifica se o veículo pode percorrer determinada distância
        com a autonomia atual.
        """
        return distancia_km <= self.autonomia_km
    
    def validar_autonomia(self, distancia_km: float):
        """
        Retorna (ok, mensagem) para verificação de autonomia.
        """
        if distancia_km > self.autonomia_km:
            excesso = distancia_km - self.autonomia_km
            return False, f"Excede autonomia disponível em {excesso:.2f} km"
        return True, "OK"
    
    def __repr__(self) -> str:
        return (
            f"Veiculo(id='{self.id_veiculo}', "
            f"capacidade={self.capacidade_kg}kg, "
            f"autonomia={self.autonomia_km}km)"
        )

@dataclass(frozen=True)
class Base:
    """
    Representa a base/hospital central de onde partem as entregas.
    Esta classe é usada como ponto de partida e retorno das rotas.
    """
    nome: str
    localizacao: Tuple[float, float]    
    
    def __repr__(self):        
        return f"Base(nome='{self.nome}', localizacao={self.localizacao})"

@dataclass
class Rota:
    """
    Representa uma rota atribuída a um veículo.

    A rota contém:
    - um veículo
    - uma sequência de entregas
    - a sequência de coordenadas visitadas (incluindo base, se aplicável)
    - métricas agregadas (distância e tempo total)

    A carga total é derivada das entregas associadas.
    """
    veiculo: Veiculo
    entregas: List[Entrega]
    sequencia: List[Tuple[float, float]] = field(default_factory=list)
    distancia_total_km: float = 0.0
    tempo_total_min: float = 0.0

    @property
    def carga_total_kg(self) -> float:
        """
        Carga total da rota (soma dos pesos das entregas).
        """
        return sum(e.peso_kg for e in self.entregas)
    
    def validar_carga_rota(self):
        """
        Valida a carga total da rota usando o validador do veículo.
        Retorna: (ok: bool, mensagem: str)
        """
        return self.veiculo.validar_carga(self.carga_total_kg)

    def validar_autonomia_rota(self):
        """
        Valida a distância total da rota usando o validador do veículo.
        Retorna: (ok: bool, mensagem: str)
        """
        return self.veiculo.validar_autonomia(self.distancia_total_km)
    
    def is_valida(self) -> bool:
        """
        Rota é válida se respeita:
        - capacidade de carga
        - autonomia do veículo
        """
        carga_ok, _ = self.validar_carga_rota()
        autonomia_ok, _ = self.validar_autonomia_rota()
        return carga_ok and autonomia_ok
    
    def custo_total(self) -> float:
        """
        Calcula o custo total da rota.

        Componentes:
        - custo operacional por distância (distância * custo_por_km)
        - penalidade por criticidade das entregas

        Obs.: Penalidades adicionais (atraso, janelas de tempo, etc.)
        podem ser tratadas no algoritmo genético, mas este método já
        fornece um custo consistente para análises simples e testes.
        """
        custo_distancia = self.distancia_total_km * self.veiculo.custo_por_km
        penalidade_prioridade = sum(
            entrega.prioridade.peso_penalidade() for entrega in self.entregas
        )
        return custo_distancia + penalidade_prioridade
    
    def __repr__(self) -> str:
        status = "✓" if self.is_valida() else "✗"
        return (
            f"Rota[{self.veiculo.id_veiculo}] {status}: "
            f"{len(self.entregas)} entregas, "
            f"{self.distancia_total_km:.1f}km, "
            f"{self.carga_total_kg:.1f}kg"
        )