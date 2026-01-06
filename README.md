# tech-challenge-fiap-otimizacao-rotas
Otimização de Rotas para Distribuição de Medicamentos e Insumos

# Na pasta onde está o código original
mkdir -p src/{core,models,llm,visualization,utils}
mkdir -p data/resultados tests notebooks docs

# Criar __init__.py em todas as pastas
touch src/__init__.py
touch src/core/__init__.py
touch src/models/__init__.py
touch src/llm/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py

# Mover arquivos originais
mv genetic_algorithm.py src/core/
mv tsp.py src/core/
mv draw_functions.py src/visualization/
mv demo_*.py src/core/  # demos são opcionais
mv benchmark_att48.py src/core/

# Criar ambiente virtual
python -m venv venv

# Ativar
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalar
pip install -r requirements.txt

# Configurar OpenAI
cp .env.example .env
# Edite .env e coloque sua chave da OpenAI