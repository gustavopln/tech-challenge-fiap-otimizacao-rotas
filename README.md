# Tech Challenge – Otimização de Rotas Médicas (FIAP)

Este projeto faz parte do **Tech Challenge – Fase 2** da pós-graduação em **IA para Devs (FIAP)** e tem como objetivo desenvolver um sistema de **otimização de rotas para distribuição de medicamentos e insumos médicos**, utilizando **Algoritmos Genéticos** e **LLMs para geração de relatórios e instruções**.

O problema é inspirado no **Caixeiro Viajante Médico**, considerando restrições realistas do contexto hospitalar, como prioridade de entregas, capacidade dos veículos e autonomia.

---

## 🧠 Objetivos do Projeto

- Resolver o problema de otimização de rotas médicas (TSP / VRP)
- Implementar Algoritmos Genéticos com restrições reais
- Modelar entidades do domínio hospitalar
- Visualizar rotas otimizadas
- Integrar LLMs para geração de relatórios e instruções
- Garantir qualidade do código com testes automatizados

---

## 📁 Estrutura do Projeto

otimizacao-rotas-medicas/

```text
.
├── data/
│   ├── entregas_exemplo.json
│   └── resultados/
│
├── docs/
│   └── relatorio_tecnico.md
│
├── notebooks/
│   └── experimentos.ipynb
│
├── src/
│   ├── core/
│   │   ├── genetic_algorithm.py
│   │   ├── medical_genetic_algorithm.py
│   │   └── tsp.py
│   │
│   ├── models/
│   │   └── models.py
│   │
│   ├── visualization/
│   │   └── draw_functions.py
│   │
│   └── llm/
│
├── tests/
│   └── test_models.py
│
├── quick_start.py
├── environment.yml
├── requirements.txt
├── pytest.ini
├── .env
└── README.md
```

---

## 🛠️ Configuração do Ambiente

### 1️⃣ Instalar o Miniconda (se necessário)

Baixe em:  
https://www.anaconda.com/docs/getting-started/miniconda/main

---

### 2️⃣ Criar e ativar o ambiente Conda

Na raiz do projeto:

```bash
conda env create -f environment.yml
conda activate fiap_tsp
```

### Verifique a versão do Python:

python --version
```bash
python --version
```

### Saída esperada
Python 3.9.19

### 3️⃣ Inicializar o Conda no Git Bash (apenas uma vez)
```bash
conda init bash
```

### Feche o Git Bash complementament:
```bash
exit
```

### Abra um novo Git Bash, navegue até o projeto e ative novamente:
```bash
conda activate fiap_tsp
```


### 🔐 Configuração da OpenAI - API Key

#### Crie o arquivo `.env` a partir do exemplo:
```bash
cp .env.example .env
```

Edite o .env e informe sua chave da OpenAI:

OPENAI_API_KEY=sua-chave-aqui
OPENAI_MODEL=gpt-4o-mini


### ▶️ Executando o Projeto

#### Para gerar dados de exemplo e rodar o pipeline inicial:

```bash
python quick_start.py
```

### 🧪 Executando os Testes Automatizados

#### O projeto utiliza pytest para validação das entidades de domínio.
```bash
PYTHONPATH=. pytest -v
```

#### Estrutura de testes atual

- **Criação de entidades**
  - Entrega
  - Veículo
  - Rota

- **Validação de restrições**
  - Capacidade de carga
  - Autonomia do veículo

- **Penalidade por prioridade médica**

- **Cálculo de custo da rota**

#### Os testes garantem que o domínio esteja correto antes da integração com o Algoritmo Genético.

### 📊 Tecnologias Utilizadas

* Python 3.9
* Conda
* Pytest
* Algoritmos Genéticos
* Pygame (visualização)
* OpenAI API (LLMs)
* Jupyter Notebook (experimentos)

### 📄 Documentação e Relatório

* O relatório técnico está em:
`docs/relatorio_tecnico.md`

* Experimentos e análises estão em:
`notebooks/experimentos.ipynb`