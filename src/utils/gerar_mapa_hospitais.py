import pandas as pd
import folium

# 1. Carregar os dados do CSV
try:
    df = pd.read_csv('data/hospitais_df.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'hospitais_df.csv' não foi encontrado. Certifique-se de salvá-lo na mesma pasta.")
    exit()

# 2. Criar o mapa centralizado em Brasília
# Coordenadas centrais aproximadas de Brasília (Plano Piloto)
mapa_df = folium.Map(location=[-15.7942, -47.8822], zoom_start=11)

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

# 4. Salvar o mapa em um arquivo HTML
mapa_df.save('data/mapa_hospitais_brasilia.html')

print("Sucesso! O mapa foi gerado e salvo como 'mapa_hospitais_brasilia.html'.")
print("Abra esse arquivo no seu navegador para visualizar o resultado.")