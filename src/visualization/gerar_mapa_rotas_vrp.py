import folium
from typing import List

from src.models.base_default import BASE_PADRAO
from src.models.models import Rota
from src.core.medical_genetic_algorithm import (
    carregar_entregas_csv,
    carregar_veiculos_csv,
    GAConfig,
    executar_ga_vrp,
)


def gerar_mapa_rotas_vrp(
    rotas: List[Rota],
    nome_arquivo: str = "data/mapa_rotas_vrp.html",
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
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
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
    print(f"Mapa de rotas VRP salvo em: {nome_arquivo}")
    print("Ative/desative as rotas dos veículos na legenda do mapa.")


if __name__ == "__main__":
    """
    Geração direta do mapa a partir da melhor solução do VRP.

    - Usa entregas.csv e veiculos.csv
    - Usa o modo semeado com rotas_iniciais.csv
    """

    base = BASE_PADRAO
    entregas = carregar_entregas_csv("data/entregas.csv")
    veiculos = carregar_veiculos_csv("data/veiculos.csv")

    config = GAConfig(
        tamanho_populacao=80,
        geracoes=80,
        taxa_mutacao=0.2,
        elitismo=0.1,
    )

    rotas, historico = executar_ga_vrp(
        entregas=entregas,
        veiculos=veiculos,
        base=base,
        config=config,
        usar_rotas_iniciais=True,
        caminho_rotas_iniciais="data/rotas_iniciais.csv",
    )

    print(f"Melhor fitness final (VRP): {historico[-1]:.2f}")

    gerar_mapa_rotas_vrp(rotas, nome_arquivo="data/mapa_rotas_vrp.html")
