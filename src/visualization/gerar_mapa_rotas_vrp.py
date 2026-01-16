import folium
from typing import List

from src.models.base_default import BASE_PADRAO
from src.models.models import Rota


def gerar_mapa_rotas_vrp(
    rotas: List[Rota],
    nome_arquivo: str = "data/resultados/mapa_rotas_vrp.html",
) -> None:
    """
    Gera um mapa Folium com as rotas do VRP:

    - Cada veículo em uma camada (FeatureGroup) separada
    - Rotas (linhas) e paradas (pontos numerados) aparecem
      somente quando a camada do veículo é ativada na legenda
    - Base sempre visível
    """

    # Centraliza o mapa na BASE
    base_lat, base_lng = BASE_PADRAO.localizacao
    mapa = folium.Map(location=[base_lat, base_lng], zoom_start=11)

    # ---------- BASE (sempre visível) ----------
    folium.Marker(
        location=[base_lat, base_lng],
        popup=BASE_PADRAO.nome,
        tooltip=BASE_PADRAO.nome,
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(mapa)

    folium.CircleMarker(
        location=[base_lat, base_lng],
        radius=8,
        color="red",
        fill=True,
        fill_opacity=0.5,
    ).add_to(mapa)

    # Paleta de cores para diferenciar visualmente os veículos
    cores = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "darkgreen",
        "cadetblue",
        "yellow",
        "darkblue",
    ]    

    # ---------- ROTAS POR VEÍCULO (em camadas separadas) ----------
    for idx_veic, rota in enumerate(rotas):
        cor = cores[idx_veic % len(cores)]
        veiculo_id = str(rota.veiculo.id_veiculo)

        # Cria uma camada específica para este veículo
        fg = folium.FeatureGroup(
            name=f"Veículo {veiculo_id}",
            show=False,  # começa oculto
        )

        coords = rota.sequencia  # lista de (lat, lng)

        # Linha da rota
        folium.PolyLine(
            locations=[(lat, lng) for (lat, lng) in coords],
            color=cor,
            weight=4,
            opacity=0.8,
            tooltip=f"Rota do veículo {veiculo_id}",
        ).add_to(fg)

        # Paradas numeradas (exclui base inicial/final se estiver em coords)
        for ordem_parada, (lat, lng) in enumerate(coords[1:-1], start=1):
            folium.CircleMarker(
                location=[lat, lng],
                radius=5,
                color=cor,
                fill=True,
                fill_opacity=0.9,
                popup=(
                    f"Veículo: {veiculo_id}<br>"
                    f"Parada: {ordem_parada}"
                ),
                tooltip=f"{veiculo_id} - parada {ordem_parada}",
            ).add_to(fg)

        fg.add_to(mapa)

    # Controle de camadas: permite ativar/desativar cada veículo
    folium.LayerControl(collapsed=False).add_to(mapa)

    # Salva o mapa
    mapa.save(nome_arquivo)    
    print(f"\nMapa das rotas VRP gerado em: {nome_arquivo}")
    print("Ative/desative as rotas dos veículos na legenda do mapa.")