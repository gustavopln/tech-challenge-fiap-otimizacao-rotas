"""
Modelos de dados para o sistema de otimização de rotas médicas
"""
from dataclasses import dataclass, field
from typing import Tuple, List
from enum import Enum

class PrioridadeEntrega(Enum):
    """Níveis de prioridade para entregas"""
    CRITICA = 1      # Medicamentos críticos, emergências
    ALTA = 2         # Medicamentos controlados, tempo-sensíveis
    MEDIA = 3        # Insumos importantes
    BAIXA = 4        # Insumos regulares

    def peso_penalidade(self) -> float:
        """
        Peso usado na função fitness.
        Quanto maior a criticidade da entrega, maior o impacto negativo no fitness em caso de violações.
        """
        return {
            PrioridadeEntrega.CRITICA: 10.0,
            PrioridadeEntrega.ALTA: 5.0,
            PrioridadeEntrega.MEDIA: 2.0,
            PrioridadeEntrega.BAIXA: 1.0,
        }[self]
    

@dataclass(frozen=True)
class Entrega:
    """Representa um ponto de entrega"""
    id: int
    nome: str
    localizacao: Tuple[float, float]  # (x, y) ou (lat, lng)    
    prioridade: PrioridadeEntrega
    peso_kg: float
    tempo_estimado_entrega_min: int = 10  # Tempo médio de parada
    janela_inicio: int = 0  # Hora início (em minutos desde 00:00)
    janela_fim: int = 1440  # Hora fim (padrão: 24h = 1440 min)

    def penalidade_prioridade(self) -> float:
        """Calcula penalidade com base na prioridade"""
        return self.prioridade.peso_penalidade()
    
    def __repr__(self):        
        return f"Entrega({self.id}, {self.nome}, P{self.prioridade.value})"

@dataclass(frozen=True)
class Veiculo:
    """Representa um veículo de entrega"""
    id: str
    capacidade_kg: float
    autonomia_km: float
    velocidade_media_kmh: float = 40.0
    custo_por_km: float = 2.50
    
    def tempo_viagem_minutos(self, distancia_km: float) -> float:
        """Calcula tempo de viagem em minutos"""
        return (distancia_km / self.velocidade_media_kmh) * 60
    
    def pode_carregar(self, peso_kg: float) -> bool:
        """Verifica se pode carregar determinado peso"""
        return peso_kg <= self.capacidade_kg
    
    def pode_percorrer(self, distancia_km: float) -> bool:
        """Verifica se pode percorrer determinada distância"""
        return distancia_km <= self.autonomia_km

@dataclass(frozen=True)
class Base:
    """Representa a base/hospital de onde partem as entregas"""
    nome: str
    localizacao: Tuple[float, float]    
    
    def __repr__(self):
        return f"Base({self.nome})"

@dataclass
class Rota:
    """
    Representa uma rota final atribuída a um veículo.
    Essa clase NÃO é usada como cromossomo no GA.
    """
    veiculo: Veiculo
    entregas: List[Entrega]
    distancia_total_km: float
    tempo_total_min: float
    sequencia: List[Tuple[float, float]]  = field(default_factory=list)  # Sequência de coordenadas

    @property
    def carga_total_kg(self) -> float:
        """Calcula carga total da rota"""
        return sum(e.peso_kg for e in self.entregas)
    
    def is_valida(self) -> bool:
        """Verifica se a rota respeita as restrições do veículo"""
        return (
            self.carga_total_kg <= self.veiculo.capacidade_kg
            and self.distancia_total_km <= self.veiculo.autonomia_km
        )
    
    def custo_total(self) -> float:
        """Calcula custo total da rota considerando:
        - custo operacional por distância
        - penalidade por criticidade das entregas
        """
        custo_distancia = self.distancia_total_km * self.veiculo.custo_por_km
        penalidade_prioridade = sum(
            entrega.prioridade.peso_penalidade() for entrega in self.entregas
        )

        return custo_distancia + penalidade_prioridade
    
    def __repr__(self):
        status = "✓" if self.is_valida() else "✗"
        return (
            f"Rota[{self.veiculo.id}] {status}: "
            f"{len(self.entregas)} entregas, "
            f"{self.distancia_total_km:.1f} km, "
            f"{self.carga_total_kg:.1f} kg"
        )