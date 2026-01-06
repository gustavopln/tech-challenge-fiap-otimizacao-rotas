"""
Modelos de dados para o sistema de otimização de rotas médicas
"""
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum

class PrioridadeEntrega(Enum):
    """Níveis de prioridade para entregas"""
    CRITICA = 1      # Medicamentos críticos, emergências
    ALTA = 2         # Medicamentos controlados, tempo-sensíveis
    MEDIA = 3        # Insumos importantes
    BAIXA = 4        # Insumos regulares

@dataclass
class Entrega:
    """Representa um ponto de entrega"""
    id: int
    localizacao: Tuple[float, float]  # (x, y) ou (lat, lng)
    nome: str
    prioridade: PrioridadeEntrega
    peso_kg: float
    tempo_estimado_entrega_min: int = 10  # Tempo médio de parada
    janela_inicio: int = 0  # Hora início (em minutos desde 00:00)
    janela_fim: int = 1440  # Hora fim (padrão: 24h = 1440 min)
    
    def __repr__(self):
        return f"Entrega({self.id}, {self.nome}, P{self.prioridade.value})"

@dataclass
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

@dataclass
class Base:
    """Representa a base/hospital de onde partem as entregas"""
    localizacao: Tuple[float, float]
    nome: str = "Hospital Base"
    
    def __repr__(self):
        return f"Base({self.nome})"

@dataclass
class Rota:
    """Representa uma rota completa de um veículo"""
    veiculo: Veiculo
    entregas: List[Entrega]
    sequencia: List[Tuple[float, float]]  # Sequência de coordenadas
    distancia_total_km: float = 0.0
    tempo_total_min: float = 0.0
    carga_total_kg: float = 0.0
    
    def is_valida(self) -> bool:
        """Verifica se a rota respeita as restrições do veículo"""
        return (self.carga_total_kg <= self.veiculo.capacidade_kg and 
                self.distancia_total_km <= self.veiculo.autonomia_km)
    
    def custo_total(self) -> float:
        """Calcula custo total da rota"""
        return self.distancia_total_km * self.veiculo.custo_por_km
    
    def __repr__(self):
        status = "✓" if self.is_valida() else "✗"
        return (f"Rota[{self.veiculo.id}] {status}: "
                f"{len(self.entregas)} entregas, "
                f"{self.distancia_total_km:.1f}km, "
                f"{self.carga_total_kg:.1f}kg")

# Exemplo de uso
if __name__ == "__main__":
    # Criar base
    base = Base(localizacao=(0, 0), nome="Hospital Universitário")
    
    # Criar entregas
    entregas = [
        Entrega(1, (10, 20), "UBS Centro", PrioridadeEntrega.CRITICA, 5.0),
        Entrega(2, (30, 40), "Clínica Norte", PrioridadeEntrega.ALTA, 10.0),
        Entrega(3, (50, 10), "PSF Sul", PrioridadeEntrega.MEDIA, 15.0),
    ]
    
    # Criar veículo
    veiculo = Veiculo("V1", capacidade_kg=50.0, autonomia_km=100.0)
    
    # Criar rota
    rota = Rota(
        veiculo=veiculo,
        entregas=entregas,
        sequencia=[base.localizacao] + [e.localizacao for e in entregas],
        distancia_total_km=75.0,
        carga_total_kg=30.0
    )
    
    print(f"Base: {base}")
    print(f"\nVeículo: {veiculo}")
    print(f"\nEntregas:")
    for entrega in entregas:
        print(f"  {entrega}")
    print(f"\n{rota}")
    print(f"Rota válida? {rota.is_valida()}")
    print(f"Custo: R$ {rota.custo_total():.2f}")
