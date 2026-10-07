# Rubrica do LLM-as-a-Judge — v1.1

**Projeto:** JurisTCU — comparação de RAG, GraphRAG e abordagem híbrida  
**Unidade de avaliação:** uma PERGUNTA, o CONTEXTO efetivamente recebido pelo gerador e sua RESPOSTA  
**Juiz do piloto:** Qwen via Groq  
**Status:** versão revisada após auditoria exploratória do piloto; exige reteste antes do congelamento experimental.

## 1. Objetivo e limites

Avaliar, de maneira cega em relação à arquitetura e ao gerador, a qualidade de respostas baseadas em evidências recuperadas do JurisTCU. O juiz trabalha **exclusivamente** com a PERGUNTA, o CONTEXTO e a RESPOSTA, sem consultar fontes externas.

As notas de correção e fidelidade indicam adequação às **evidências fornecidas**, e não uma verificação independente da verdade jurídica ou da vigência normativa.

## 2. Relevância (`relevance`)

Avalia em que medida a resposta aborda o que foi perguntado, sem fugir do tema.

| Nota | Interpretação |
|---|---|
| 5 | Responde diretamente ao que foi solicitado. |
| 4 | Responde principalmente à pergunta, com pequeno desvio ou foco imperfeito. |
| 3 | Responde parcialmente, com lacunas perceptíveis de foco. |
| 2 | Trata de aspectos próximos, mas pouco atende ao pedido. |
| 1 | Não responde de modo útil à pergunta. |

## 3. Completude (`completeness`) — definição revisada na v1.1

**Definição operacional:** grau em que a resposta cobre os pontos **materialmente necessários** para responder à pergunta, à luz das evidências disponíveis no contexto.

A completude **não** mede a cobertura de todos os documentos recuperados. Um detalhe só deve provocar perda de pontos quando sua ausência prejudicar a resposta ao que foi efetivamente perguntado.

| Nota | Interpretação |
|---|---|
| 5 | Contém todos os pontos necessários, incluindo condições, exceções e qualificações essenciais disponíveis no contexto. Pode ser curta. |
| 4 | Cobre o núcleo da pergunta, mas omite uma qualificação necessária de impacto limitado. A justificativa deve identificar a omissão e explicar seu efeito. |
| 3 | Omite uma ou mais partes materiais importantes da resposta, prejudicando sua utilidade. |
| 2 | Cobre apenas um fragmento do conteúdo necessário. |
| 1 | Praticamente não cobre o que seria necessário para responder à pergunta. |

### Procedimento obrigatório de julgamento

1. Determine o **núcleo da pergunta** (qual informação está sendo solicitada?).
2. Determine quais informações no **contexto** são indispensáveis para responder a esse núcleo com suas condições e exceções relevantes.
3. Confira se essas informações aparecem na **resposta**.
4. **Não penalize** a ausência de detalhes acessórios, exemplos opcionais, contextualização histórica, valores financeiros periféricos ou dados de outros documentos quando não forem necessários à pergunta.
5. Para uma nota abaixo de 5, documente uma **omissão material concreta** e explique **por que ela faz falta para responder àquela pergunta**. Se não houver tal omissão, use 5.

**Exemplo metodológico inspirado na auditoria:** quando uma consulta pede a *modalidade de licitação* aplicável, identificar a modalidade e ressalvas essenciais pode bastar para nota 5. Omissões sobre características contratuais ou composição do valor da remuneração não reduzem a nota, a menos que a pergunta também solicite esses aspectos. Por outro lado, quando a pergunta pede os requisitos ou consequências da cessão contratual, omitir a restrição à transferência de posição contratual poderá ser material. **Esse exemplo não deve ser inserido no prompt operacional do juiz.**

### Diferenças entre métricas

- Uma resposta pode ser **fiel e correta**, mas incompleta se omitir uma condição essencial.
- Uma resposta pode abordar todos os pontos pedidos, mas conter erro ou extrapolação; nesse caso, a falha deve ser registrada em `correctness`/claims sem compensação na nota de completude.
- **Resposta longa não é sinônimo de resposta completa.**

## 4. Correção (`correctness`)

Avalia se as afirmações representam corretamente as evidências fornecidas, sem contradições ou inferências não autorizadas.

| Nota | Interpretação |
|---|---|
| 5 | Sem erros identificáveis em relação ao contexto. |
| 4 | Pequena imprecisão sem alterar a conclusão central. |
| 3 | Imprecisão ou erro relevante, mas a resposta conserva alguma utilidade. |
| 2 | Erros substanciais comprometem a interpretação. |
| 1 | Informação essencialmente errada ou incompatível com o contexto. |

Não se deve usar conhecimento jurídico externo para corrigir o material recuperado.

## 5. Fidelidade e claims

Uma *claim* é uma afirmação substantiva, factual ou jurídica, verificável à luz do contexto. Não contam como claims: cumprimentos, avisos de insuficiência, frases sobre o próprio processo de avaliação e citações sem afirmação.

Classifique cada claim em uma de duas categorias:

- **Supported:** sustentada diretamente ou por paráfrase fiel pelo contexto.
- **Unsupported:** não sustentada, extrapolada ou contradita pelo contexto.

Regras de consistência:

```text
total_claims = supported_claims + unsupported_claims
```

Métricas calculadas no Python:

\[
Faithfulness = \frac{supported\_claims}{total\_claims}
\]

\[
UnsupportedClaimRate = \frac{unsupported\_claims}{total\_claims}
\]

Se `total_claims = 0`, ambas devem ser `null`, não zero nem um. Quando toda claim pertence a uma das duas categorias, as métricas são complementares; não tratá-las como duas evidências estatísticas independentes.

A presença de `[DOC_...]` na resposta não prova suporte: é preciso verificar o texto do contexto identificado.

## 6. Suficiência do contexto e decisão de abstenção

- `context_sufficient=true`: há evidências suficientes para uma resposta substantiva adequada ao núcleo da pergunta.
- `context_sufficient=false`: faltam evidências para responder de maneira substantiva e adequada ao núcleo da pergunta.
- `response_abstained=true`: a resposta não traz solução substantiva e explicitamente deixa de responder por insuficiência de evidências.
- `response_abstained=false`: existe resposta substantiva, mesmo quando parcial ou acompanhada de ressalva sobre limites do contexto.

Se `response_abstained=true`, as notas `relevance`, `completeness` e `correctness` ficam `null`, e as três contagens de claims ficam em zero. A decisão de se abster é tratada separadamente.

O Python, **não o juiz**, calcula:

```python
appropriate_abstention = (
    (not context_sufficient and response_abstained)
    or (context_sufficient and not response_abstained)
)
```

Assim, responder quando há evidência e abster-se quando ela falta são ambos comportamentos apropriados. Os outros dois cenários são inadequados.

## 7. Contrato JSON do juiz (não alterar)

O juiz devolve somente estes campos, que são exigidos pelo avaliador Python:

```json
{
  "total_claims": 0,
  "supported_claims": 0,
  "unsupported_claims": 0,
  "relevance": null,
  "completeness": null,
  "correctness": null,
  "context_sufficient": false,
  "response_abstained": true,
  "justification": {
    "relevance": "Motivo breve.",
    "completeness": "Motivo breve.",
    "correctness": "Motivo breve.",
    "unsupported_claims": []
  }
}
```

O exemplo representa uma **abstenção**, não valores universais. Em respostas substantivas, as três notas são inteiros entre 1 e 5. O juiz não retorna `faithfulness`, `unsupported_claim_rate` nem `appropriate_abstention` — calculados no programa Python.

## 8. Registro da alteração e controle experimental

**v1.0:** versão usada na primeira avaliação exploratória de 18 respostas do piloto (6 consultas × 3 arquiteturas). Na consulta 51, as três arquiteturas receberam nota 4 em completude por não incluir detalhes reconhecidos nas justificativas como secundários à pergunta.

**v1.1:** esclarece que somente omissões **materialmente necessárias para responder à pergunta** podem reduzir `completeness`, e exige que a justificativa identifique a omissão e seu efeito. As demais regras e o contrato JSON foram mantidos compatíveis com o programa de avaliação.

A auditoria da consulta 51 motivou a revisão, **mas não garante que as próximas notas serão 5**. O juiz deve continuar julgando cada resposta individualmente e poderá atribuir qualquer nota quando houver evidências para isso.

**Protocolo recomendado:**

1. Preserve os arquivos de prompt, rubrica e resultados originais v1.0 para rastreabilidade.
2. Reavalie as 18 respostas do piloto com o prompt v1.1, sem regenerar as respostas do Gemini.
3. Compare nota a nota por `(query_id, arquitetura, modelo_gerador)`, investigando variações inesperadas em todas as métricas.
4. Faça inspeção humana de uma seleção diversificada de casos, inclusive notas 5, e não só dos casos de baixa nota.
5. Congele uma única versão do juiz antes da avaliação principal e documente qualquer revisão posterior.

**Limitações:** o juiz pode demonstrar permissividade, viés de modelo ou dificuldade de separar claims. Notas máximas uniformes não provam ausência de erros; é recomendável validar manualmente uma amostra, especialmente no domínio jurídico.
