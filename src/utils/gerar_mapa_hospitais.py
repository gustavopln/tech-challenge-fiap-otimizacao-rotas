import pandas as pd
import folium

# Coordenadas da base (Almoxarifado Central)
BASE_LAT = -15.8140447
BASE_LNG = -47.9659546
BASE_NOME = "Almoxarifado Central da Secretaria de Saúde do DF"

# 1. Carregar os dados do CSV
try:
    df = pd.read_csv('data/hospitais_df.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'hospitais_df.csv' não foi encontrado. Certifique-se de salvá-lo na mesma pasta.")
    raise SystemExit(1)

# 2. Criar o mapa centralizado em Brasília (pode ser no plano piloto ou na base)
mapa_df = folium.Map(location=[BASE_LAT, BASE_LNG], zoom_start=11)

# 3. Adicionar os marcadores ao mapa
for index, row in df.iterrows():
    # Definindo a cor do marcador baseado no tipo (opcional)
    # Aqui vamos usar azul para hospitais e verde para UBS/UPAs de forma simples
    cor = 'blue'
    if 'UBS' in row['Nome'] or 'UPA' in row['Nome']:
        cor = 'green'
    
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Nome'],  # Texto que aparece ao clicar
        tooltip=row['Nome'], # Texto que aparece ao passar o mouse
        icon=folium.Icon(color=cor, icon='plus', prefix='fa') # Ícone de cruz médica
    ).add_to(mapa_df)

# 4. Adicionar marcador especial para a BASE
folium.Marker(
    location=[BASE_LAT, BASE_LNG],
    popup=BASE_NOME,
    tooltip=BASE_NOME,
    icon=folium.Icon(color="red", icon="home", prefix="fa"),  # base em vermelho com ícone de "home"
).add_to(mapa_df)

# 4.1. Adicionar um círculo em volta da base para destacá-la ainda mais
folium.CircleMarker(
    location=[BASE_LAT, BASE_LNG],
    radius=8,
    color="red",
    fill=True,
    fill_opacity=0.5,
).add_to(mapa_df)

# 5. Salvar o mapa em um arquivo HTML
mapa_df.save('data/mapa_hospitais_brasilia.html')

print("Sucesso! O mapa foi gerado e salvo como 'mapa_hospitais_brasilia.html'.")
print("Abra esse arquivo no seu navegador para visualizar o resultado.")