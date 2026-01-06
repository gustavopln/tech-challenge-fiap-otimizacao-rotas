# tech-challenge-fiap-otimizacao-rotas
Otimização de Rotas para Distribuição de Medicamentos e Insumos

# 1. Instale Miniconda (se não tiver)
Baixe em: https://www.anaconda.com/docs/getting-started/miniconda/main

# Navegue até a pasta do seu projeto
cd "../../.."

# Crie o ambiente
conda env create -f environment.yml

# Ative
conda activate fiap_tsp

# Verifique
python --version

# 1. Inicializar Conda
conda init bash

# 2. FECHE o Git Bash atual (importante!)
exit

# 3. Abra um NOVO Git Bash

# 4. Navegue até seu projeto
cd "../../.."

# 5. Ative o ambiente
conda activate fiap_tsp

# 6. Verifique
python --version  # Deve mostrar Python 3.9.19

# Configurar OpenAI
cp .env.example .env
# Edite .env e coloque sua chave da OpenAI