# Rubrica de Avaliação da Geração

## Relevância

Avalia se a resposta atende diretamente à pergunta.

- 1: não responde à pergunta ou é majoritariamente irrelevante.
- 2: responde apenas minimamente.
- 3: responde parcialmente, deixando aspectos importantes sem tratamento.
- 4: responde adequadamente, com pequenas omissões.
- 5: responde diretamente e de forma plenamente pertinente.

## Completude

Avalia a cobertura das informações relevantes disponíveis no contexto.

- 1: praticamente todas as informações importantes foram omitidas.
- 2: apenas pequena parte das evidências relevantes foi utilizada.
- 3: cobertura parcial, com omissões importantes.
- 4: cobre quase todas as informações importantes.
- 5: utiliza adequadamente todas as informações relevantes disponíveis.

A completude deve ser avaliada apenas em relação ao contexto fornecido.

## Correção

Avalia se a resposta interpreta corretamente as evidências fornecidas.

- 1: contradiz ou distorce gravemente o contexto.
- 2: contém vários erros relevantes.
- 3: parcialmente correta, com erros ou distorções.
- 4: essencialmente correta, com pequenas imprecisões.
- 5: representa corretamente as evidências fornecidas.

Não utilizar conhecimento externo para completar ou corrigir o contexto.

## Faithfulness

A resposta deve ser decomposta em afirmações factuais verificáveis.

Cada afirmação deverá ser classificada como:

- suportada pelo contexto;
- não suportada pelo contexto.

Faithfulness:

supported_claims / total_claims

## Unsupported Claim Rate

unsupported_claims / total_claims

## Abstenção

O avaliador deverá identificar:

- se o contexto é suficiente para responder à pergunta;
- se a resposta se absteve de responder;
- se o comportamento de abstenção foi adequado.

A abstenção é adequada quando:

1. o contexto é insuficiente e a resposta reconhece essa limitação; ou
2. o contexto é suficiente e a resposta apresenta uma resposta substancial.

## Avaliação cega

O juiz não receberá informações sobre:

- arquitetura de recuperação;
- modelo que gerou a resposta;
- identificação experimental do sistema.

A avaliação será realizada somente com base em:

1. pergunta;
2. contexto;
3. resposta.