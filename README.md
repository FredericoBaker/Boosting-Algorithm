## Implementação Boosting

Este repositório implementa o algoritmo de Boosting, uma técnica de ensemble learning que combina múltiplos modelos fracos para formar um modelo forte, melhorando a precisão das previsões de classificação. Modelos fracos têm uma taxa de acerto ligeiramente superior ao acaso, e o Boosting ajusta iterativamente esses modelos para focar nos erros dos modelos anteriores, reduzindo o viés e mantendo a variância baixa.

### Metodologia
#### Algoritmo de Boosting
O algoritmo de Boosting foi implementado em Python e funciona da seguinte forma:

1. Inicialização dos pesos das observações.
2. Loop:
   - Seleciona o modelo que minimiza o erro empírico na iteração.
   - Calcula o valor de alpha para a iteração.
   - Atualiza os pesos das observações.
3. O loop continua até que o erro empírico seja próximo de zero.

#### Classe DecisionStump
Implementa um modelo fraco que percorre todas as features e thresholds para definir o decision stump com menor erro na iteração. O erro considera os pesos das observações e é normalizado para que o somatório seja igual a 1. A classe também determina se a melhor previsão é uma classe positiva à direita ou à esquerda do decision stump.

#### Classe AdaBoost
Cria múltiplos decision stumps, um por iteração, e calcula o alpha, que indica a importância da previsão de um decision stump no modelo final. O alpha é calculado pela fórmula:
!(Fórmula Alpha)[./formula_alpha.png]
A classe escolhe o decision stump com menor erro, atualiza os pesos das observações e calcula o alpha para o modelo.

### Resultados
Os experimentos foram realizados usando o dataset Tic-Tac-Toe Endgame. O dataset foi dividido em treino (80%) e teste (20%). A performance do algoritmo foi avaliada usando validação cruzada com 5 folds.

#### Gráfico de Erros
O gráfico a seguir mostra o erro médio e o desvio padrão para cada iteração. 

!(Gráfico de erros)[./errors_plot.png]

Observações importantes:
- Com poucas iterações, o erro é relativamente alto (~0.3).
- Com menos estimadores, o desvio padrão é maior.
- A partir de 200 estimadores, o erro de teste é consistentemente menor que o erro de treino.
- O menor erro foi alcançado com 254 iterações, com um erro de 0.0157 e acurácia de 0.984 no conjunto de teste.

#### Gráfico de Tempo de Treinamento
O gráfico a seguir mostra o tempo de treinamento para cada número de iterações. O tempo cresce linearmente com o número de iterações, indicando a importância de otimizar o número de estimadores para garantir eficiência computacional.

!(Gráfico tempo de treinamento)[./elapsed_time_plot.png]

### Conclusão
O Boosting é eficaz na redução do erro de classificação, especialmente com um número suficiente de estimadores. O modelo com 254 iterações foi escolhido como o melhor balanço entre erro e complexidade computacional, apresentando um erro de 0.0157 e uma acurácia de 0.984 no conjunto de teste.
