# Reconhecimento de Números MNIST

Um exercício simples com o objetivo principal de testar a capacidade física da RPi em executar algoritimos similares.

## Resultados

### Raspberry Pi 3B
```
input_nodes = 784
hidden_nodes = 10
output_nodes = 10
epochs = 1
```
  - Time Elapsed: 99.1s
  - Performance: 89%

```
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
epochs = 5
```
  - Time Elapsed:
  - Performance:

### i9-10900K + DDR4 3600MHz
```
input_nodes = 784
hidden_nodes = 10
output_nodes = 10
epochs = 1
```
  - Time Elapsed: 1.6s
  - Performance: 78%

```
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
epochs = 5
```
  - Time Elapsed: 31.7s
  - Performance: 97%

## Conclusão
Mesmo sendo nítida a diferença no tempo de execução entre a minha RPi e o meu i9, é impressionante a sequer capacidade de execução deste tipo de tarefa na Raspberry Pi, isso que com certeza ainda existem otimizações adicionais que podem melhorar a performance desse tipo de algorimito em SBCs.

Outra observação relevante é a volatilidade da performance quando é utilizado poucos 'epochs' e 'hidden nodes', pelo que observei existe uma variabilidade de até 10% na taxa de acerto.
