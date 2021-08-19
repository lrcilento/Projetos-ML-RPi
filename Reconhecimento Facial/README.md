# Reconhecimento Facial

Neste exercício foi desenvolvido um algoritmo de reconhecimento facial utilizando as bibliotecas 'dblib' e 'facial_recognition'. 

O dataset foi gerado a partir de cenas de um dos meus filmes favoritos: 'There Will Be Blood', com o objetivo de reconhecer e diferenciar os dois personagens principais: Daniel Plainview e Paul Sunday.

Assim como os outros exercícios, ele será executado tanto no meu computador (i9-10900K + DDR4 3600MHz) quanto na minha Raspberry Pi 3B afim de comparar os resultados.

## Resultados

```
Exemplo 1: Reconheceu apenas um personagem. (Paul Sunday)
Exemplo 2: Reconheceu apenas um personagem. (Paul Sunday)
Exemplo 3: Reconheceu ambos personagens!
Exemplo 4: Não reconheceu nenhum personagem.
Exemplo 5: Reconheceu ambos personagens!
Exemplo 6: Não reconheceu nenhum personagem.
Exemplo 7: Não reconheceu nenhum personagem.
Video Exemplo 1: Reconheceu ambos os personagens durante o vídeo todo!
Video Exemplo 2: Reconheceu apenas um personagem durante o vídeo todo. (Paul Sunday)
```

### i9-10900K + DDR4 3600MHz

A criação do arquivo '*encodings.pickle*' levou **1500s (25 minutos)** usando o método CNN e apenas **29s** usando o método HOG.
Os tempos de execução de todas as imagens foram abaixo de **0.0005s**, e dos vídeos foram, respectivamente: **257s e 230s**.

## Conclusão

Acredito que a diferença de performance em relação aos dois personagens se dá pelo fato do Dataset de um deles (Paul Sunday) ter uma maior qualidade.
