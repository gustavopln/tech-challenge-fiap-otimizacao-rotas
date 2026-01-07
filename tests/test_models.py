import pytest
from src.models.models import (
    Base,
    Entrega,
    Veiculo,    
    Rota,
    PrioridadeEntrega,
)

def test_prioridade_peso_penalidade():
    assert PrioridadeEntrega.CRITICA.peso_penalidade() > PrioridadeEntrega.ALTA.peso_penalidade()
    assert PrioridadeEntrega.ALTA.peso_penalidade() > PrioridadeEntrega.MEDIA.peso_penalidade()
    assert PrioridadeEntrega.MEDIA.peso_penalidade() > PrioridadeEntrega.BAIXA.peso_penalidade()

def test_criacao_entrega():
    entrega = Entrega(
        id=1,
        nome="UBS Vila Mariana",
        localizacao=(450, 250),
        prioridade=PrioridadeEntrega.CRITICA,
        peso_kg=5.0,
        tempo_estimado_entrega_min=15
    )
    assert entrega.id == 1
    assert entrega.nome == "UBS Vila Mariana"
    assert entrega.localizacao == (450, 250)
    assert entrega.prioridade == PrioridadeEntrega.CRITICA
    assert entrega.peso_kg == 5.0
    assert entrega.tempo_estimado_entrega_min == 15

def test_criacao_veiculo():
    veiculo = Veiculo(
        id="V1",
        capacidade_kg=50.0,
        autonomia_km=100.0,
        velocidade_media_kmh=40.0,
        custo_por_km=2.50
    )
    assert veiculo.id == "V1"
    assert veiculo.capacidade_kg == 50.0
    assert veiculo.autonomia_km == 100.0
    assert veiculo.velocidade_media_kmh == 40.0
    assert veiculo.custo_por_km == 2.50

def test_rota_valida():
    base = Base(localizacao=(400, 200), nome="Hospital Base")

    entregas = [
        Entrega(1, (10, 0), "UBS A", PrioridadeEntrega.ALTA, 10.0),
        Entrega(2, (20, 0), "UBS B", PrioridadeEntrega.MEDIA, 15.0),
    ]

    veiculo = Veiculo("V1", capacidade_kg=30.0, autonomia_km=50.0)

    rota = Rota(
        veiculo=veiculo,
        entregas=entregas,
        sequencia=[base.localizacao] + [e.localizacao for e in entregas],
        distancia_total_km=40.0,    
        tempo_total_min=60
    )

    assert rota.is_valida() == True

def test_rota_invalida_por_carga():
    base = Base(localizacao=(400, 200), nome="Hospital Base")

    entregas = [
        Entrega(1, (10, 0), "UBS A", PrioridadeEntrega.ALTA, 30.0),
        Entrega(2, (20, 0), "UBS B", PrioridadeEntrega.MEDIA, 30.0),
    ]

    veiculo = Veiculo("V1", capacidade_kg=50.0, autonomia_km=200.0)

    rota = Rota(
        veiculo=veiculo,
        entregas=entregas,
        sequencia=[base.localizacao] + [e.localizacao for e in entregas],
        distancia_total_km=40.0,        
        tempo_total_min=60
    )

    assert rota.is_valida() == False

def test_rota_invalida_por_autonomia():
    base = Base(localizacao=(0, 0), nome="Hospital Base")

    entregas = [
        Entrega(1, (10, 0), "UBS A", PrioridadeEntrega.CRITICA, 5.0),
    ]

    veiculo = Veiculo("V1", capacidade_kg=50.0, autonomia_km=20.0)      

    rota = Rota(
        veiculo=veiculo,
        entregas=entregas,
        sequencia=[base.localizacao] + [e.localizacao for e in entregas],
        distancia_total_km=30.0,        
        tempo_total_min=40
    )

    assert rota.is_valida() == False

def test_custo_rota_considera_prioridade():
    base = Base(localizacao=(0, 0), nome="Hospital Base")

    entrega_critica = Entrega(1, (10, 0), "UTI", PrioridadeEntrega.CRITICA, 5.0)
    entrega_baixa = Entrega(2, (20, 0), "Almoxarifado", PrioridadeEntrega.BAIXA, 5.0)

    veiculo = Veiculo("V1", capacidade_kg=50.0, autonomia_km=100.0, custo_por_km=2.0)

    rota_critica = Rota(
        veiculo=veiculo,
        entregas=[entrega_critica],
        sequencia=[base.localizacao, entrega_critica.localizacao],
        distancia_total_km=20.0,        
        tempo_total_min=50
    )

    rota_baixa = Rota(
        veiculo=veiculo,
        entregas=[entrega_baixa],
        sequencia=[base.localizacao, entrega_baixa.localizacao],
        distancia_total_km=20.0,        
        tempo_total_min=30
    )

    assert rota_critica.custo_total() > rota_baixa.custo_total()