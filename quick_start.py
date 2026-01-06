"""
Script de início rápido para testar o sistema de rotas médicas
Execute este arquivo para verificar se tudo está funcionando
"""
import random
import json
from pathlib import Path

# Importações do seu código original
from src.core.genetic_algorithm import calculate_distance

# Importações dos novos módulos (quando criar os arquivos)
# from src.models import Entrega, Veiculo, Base, PrioridadeEntrega
# from src.core.medical_ga import calculate_fitness_with_constraints, generate_priority_biased_population


def criar_dados_exemplo():
    """Cria um arquivo JSON com dados de exemplo para testes"""
    
    dados = {
        "base": {
            "nome": "Hospital Universitário FIAP",
            "localizacao": [400, 200],
            "endereco": "Av. Lins de Vasconcelos, 1222 - Aclimação, São Paulo"
        },
        "veiculos": [
            {
                "id": "V1",
                "tipo": "Van",
                "capacidade_kg": 50.0,
                "autonomia_km": 100.0,
                "velocidade_media_kmh": 40.0,
                "custo_por_km": 2.50
            },
            {
                "id": "V2",
                "tipo": "Moto",
                "capacidade_kg": 15.0,
                "autonomia_km": 80.0,
                "velocidade_media_kmh": 50.0,
                "custo_por_km": 1.50
            }
        ],
        "entregas": [
            {
                "id": 1,
                "nome": "UBS Vila Mariana",
                "localizacao": [450, 250],
                "prioridade": "CRITICA",
                "tipo_material": "Soro antiofídico",
                "peso_kg": 5.0,
                "tempo_entrega_min": 15,
                "observacao": "Refrigeração necessária"
            },
            {
                "id": 2,
                "nome": "Clínica São Judas",
                "localizacao": [500, 150],
                "prioridade": "ALTA",
                "tipo_material": "Insulina",
                "peso_kg": 3.0,
                "tempo_entrega_min": 10
            },
            {
                "id": 3,
                "nome": "PSF Jabaquara",
                "localizacao": [350, 300],
                "prioridade": "MEDIA",
                "tipo_material": "Vacinas",
                "peso_kg": 12.0,
                "tempo_entrega_min": 20
            },
            {
                "id": 4,
                "nome": "UPA Saúde",
                "localizacao": [550, 200],
                "prioridade": "MEDIA",
                "tipo_material": "EPIs",
                "peso_kg": 8.0,
                "tempo_entrega_min": 10
            },
            {
                "id": 5,
                "nome": "Hospital Santa Cruz",
                "localizacao": [320, 180],
                "prioridade": "ALTA",
                "tipo_material": "Hemoderivados",
                "peso_kg": 4.0,
                "tempo_entrega_min": 15,
                "observacao": "Prazo: 2 horas"
            },
            {
                "id": 6,
                "nome": "Clínica Saúde Total",
                "localizacao": [480, 320],
                "prioridade": "BAIXA",
                "tipo_material": "Material de escritório",
                "peso_kg": 10.0,
                "tempo_entrega_min": 5
            }
        ]
    }
    
    # Cria diretório data se não existir
    Path("data").mkdir(exist_ok=True)
    
    # Salva arquivo
    with open("data/entregas_exemplo.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
    
    print("✓ Arquivo 'data/entregas_exemplo.json' criado com sucesso!")
    return dados


def testar_codigo_original():
    """Testa se o código TSP original está funcionando"""
    print("\n=== TESTANDO CÓDIGO ORIGINAL ===")
    
    # Testa função de distância
    p1 = (0, 0)
    p2 = (3, 4)
    dist = calculate_distance(p1, p2)
    print(f"✓ Distância entre {p1} e {p2}: {dist:.2f} pixels")
    
    # Testa com pontos aleatórios
    pontos = [(random.randint(0, 500), random.randint(0, 500)) for _ in range(5)]
    print(f"✓ Gerados {len(pontos)} pontos aleatórios")
    
    # Calcula distância total de uma rota simples
    distancia_total = 0
    for i in range(len(pontos)):
        distancia_total += calculate_distance(pontos[i], pontos[(i+1) % len(pontos)])
    
    print(f"✓ Distância total da rota: {distancia_total:.2f} pixels")


def mostrar_proximos_passos():
    """Mostra próximos passos para o aluno"""
    print("\n" + "="*60)
    print("🎯 PRÓXIMOS PASSOS")
    print("="*60)
    
    passos = [
        {
            "num": 1,
            "titulo": "Organizar estrutura de pastas",
            "tarefas": [
                "Criar pasta src/ e subpastas (core, models, llm, visualization)",
                "Mover arquivos originais para src/core/",
                "Criar arquivo __init__.py em cada pasta"
            ]
        },
        {
            "num": 2,
            "titulo": "Implementar classes base (models.py)",
            "tarefas": [
                "Copiar código do artifact 'models.py'",
                "Testar criação de objetos Entrega, Veiculo, Base",
                "Validar que as restrições funcionam"
            ]
        },
        {
            "num": 3,
            "titulo": "Adaptar algoritmo genético",
            "tarefas": [
                "Copiar código do artifact 'medical_genetic_algorithm.py'",
                "Integrar com genetic_algorithm.py original",
                "Testar função fitness com penalidades"
            ]
        },
        {
            "num": 4,
            "titulo": "Criar visualização adaptada",
            "tarefas": [
                "Modificar tsp.py para usar novos modelos",
                "Adicionar legenda de prioridades (cores diferentes)",
                "Mostrar métricas: capacidade usada, autonomia restante"
            ]
        },
        {
            "num": 5,
            "titulo": "Começar integração com LLM",
            "tarefas": [
                "Instalar biblioteca (openai ou anthropic)",
                "Criar função para gerar instruções de rota",
                "Testar geração de relatório simples"
            ]
        }
    ]
    
    for passo in passos:
        print(f"\n📌 PASSO {passo['num']}: {passo['titulo']}")
        for i, tarefa in enumerate(passo['tarefas'], 1):
            print(f"   {i}. {tarefa}")
    
    print("\n" + "="*60)
    print("💡 DICA: Faça um commit no Git após cada passo concluído!")
    print("="*60)


def main():
    """Função principal"""
    print("="*60)
    print("🏥 SISTEMA DE OTIMIZAÇÃO DE ROTAS MÉDICAS")
    print("   FIAP - Tech Challenge Fase 2")
    print("="*60)
    
    # 1. Criar dados de exemplo
    print("\n[1/3] Criando dados de exemplo...")
    dados = criar_dados_exemplo()
    print(f"     → {len(dados['entregas'])} entregas cadastradas")
    print(f"     → {len(dados['veiculos'])} veículos disponíveis")
    
    # 2. Testar código original
    print("\n[2/3] Testando código TSP original...")
    testar_codigo_original()
    
    # 3. Mostrar próximos passos
    print("\n[3/3] Ambiente configurado!")
    mostrar_proximos_passos()
    
    print("\n✅ Setup inicial completo! Você está pronto para começar.")
    print("\n📺 Lembre-se: você precisará fazer um vídeo de 15 min no final!")


if __name__ == "__main__":
    main()
