# src/utils/relatorios_ga.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable

@dataclass
class GAEvolutionReport:
    """
    Relatório de evolução do Algoritmo Genético para UM cenário.

    Uso típico neste projeto:
      - cenário: 'VRP - população semeada'
      - historico_fitness: lista com o melhor fitness por geração
        (quanto MENOR o fitness, melhor a solução).

    Convenções:
      - historico_fitness[0]  -> melhor fitness da população inicial
      - historico_fitness[-1] -> melhor fitness após todas as gerações
    """

    nome_cenario: str
    historico_fitness: List[float]

    @property
    def num_geracoes(self) -> int:
        return len(self.historico_fitness)

    @property
    def fitness_inicial(self) -> float:
        return float(self.historico_fitness[0])

    @property
    def fitness_final(self) -> float:
        return float(self.historico_fitness[-1])

    @property
    def melhor_fitness(self) -> float:
        return float(min(self.historico_fitness))

    @property
    def geracao_melhor_fitness(self) -> int:
        """
        Retorna a geração (1-based) em que ocorreu o melhor fitness.
        Ex.: retorno 1 significa 'primeira geração observada'.
        """
        idx = min(
            range(len(self.historico_fitness)),
            key=lambda i: self.historico_fitness[i],
        )
        return idx + 1

    @property
    def melhoria_absoluta(self) -> float:
        """
        Quanto o fitness melhorou (diminuiu) do início ao fim.

        Valor positivo significa melhora (fitness_final < fitness_inicial).
        """
        return self.fitness_inicial - self.fitness_final

    @property
    def melhoria_relativa_pct(self) -> float:
        """
        Melhora percentual em relação ao fitness inicial.

        Ex.: 25.0 -> 25% de melhora.
        """
        if self.fitness_inicial == 0:
            return 0.0
        return (self.melhoria_absoluta / self.fitness_inicial) * 100.0

    # def resumo_texto(self) -> str:
    #     """
    #     Gera um resumo textual em português, pronto para ser usado no relatório técnico ou impresso no console.
    #     """
    #     linhas = [
    #         f"=== Evolução do cenário: {self.nome_cenario} ===",
    #         f"Número de gerações avaliadas: {self.num_geracoes}",
    #         f"Fitness inicial (população semeada): {self.fitness_inicial:.2f}",
    #         f"Fitness final (após evolução): {self.fitness_final:.2f}",
    #         f"Melhor fitness observado: {self.melhor_fitness:.2f}",
    #         f"Geração do melhor fitness: {self.geracao_melhor_fitness}",
    #         f"Melhoria absoluta (inicial - final): {self.melhoria_absoluta:.2f}",
    #         f"Melhoria relativa: {self.melhoria_relativa_pct:.2f}%",
    #     ]
    #     return "\n".join(linhas)


def relatorio_populacao_inicial_vs_final(
    nome_cenario: str,
    historico_fitness: List[float],
) -> str:
    """
    Gera um texto focado em comparar população inicial (semeada) vs população evoluída (filhos/mutação ao longo das gerações).

    Pressupõe que:
      - historico_fitness[0]  = melhor indivíduo da população inicial
      - historico_fitness[-1] = melhor indivíduo ao final da evolução
    """
    if not historico_fitness:
        raise ValueError("historico_fitness vazio; não há dados para o relatório.")

    report = GAEvolutionReport(nome_cenario=nome_cenario, historico_fitness=historico_fitness)

    sep = "=" * 94

    linhas = [
        sep,
        f"=== População inicial vs população evoluída ({nome_cenario}) ===",
        sep,
        f"Gerações avaliadas: {report.num_geracoes}",
        f"Fitness população inicial (geração 1): {report.fitness_inicial:.2f}",
        f"Fitness população final (geração {report.num_geracoes}): {report.fitness_final:.2f}",
        f"Melhor fitness observado: {report.melhor_fitness:.2f}",
        f"Geração do melhor fitness: {report.geracao_melhor_fitness}",
        f"Melhoria absoluta (inicial - final): {report.melhoria_absoluta:.2f}",
        f"Melhoria relativa: {report.melhoria_relativa_pct:.2f}%",
    ]

    return "\n".join(linhas)


def exportar_historico_para_csv(
    caminho: str | Path,
    historico: Iterable[float],
    nome_coluna: str = "fitness",
) -> Path:
    """
    Exporta o histórico de fitness para um CSV simples.

    Formato:
        geracao,<nome_coluna>
        1, ...
        2, ...
        ...

    Isso é útil para gerar gráficos no notebooks/experimentos.ipynb e incluir prints no relatório técnico.
    """
    caminho = Path(caminho)
    if caminho.parent and not caminho.parent.exists():
        caminho.parent.mkdir(parents=True, exist_ok=True)

    valores = list(historico)
    if not valores:
        raise ValueError("Histórico vazio; nada para exportar.")

    with caminho.open("w", newline="", encoding="utf-8") as f:
        f.write(f"geracao,{nome_coluna}\n")
        for i, val in enumerate(valores, start=1):
            f.write(f"{i},{val:.6f}\n")

    return caminho
