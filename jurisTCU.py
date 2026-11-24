from datasets import load_dataset

BASE_URL = "hf://datasets/LeandroRibeiro/JurisTCU/"

documentos = load_dataset(
    "csv",
    data_files=BASE_URL + "doc.csv",
    split="train"
)

consultas = load_dataset(
    "csv",
    data_files=BASE_URL + "query.csv",
    split="train"
)

relevancias = load_dataset(
    "csv",
    data_files=BASE_URL + "qrel.csv",
    split="train"
)

print("=" * 60)
print("DOCUMENTOS")
print(documentos)

print("\n" + "=" * 60)
print("CONSULTAS")
print(consultas)

print("\n" + "=" * 60)
print("RELEVÂNCIAS")
print(relevancias)


print("\n" + "=" * 60)
print("EXEMPLO DE DOCUMENTO")
print(documentos[0])

print("\n" + "=" * 60)
print("EXEMPLOS DE CONSULTAS")
for i in range(min(5, len(consultas))):
    print(consultas[i])

print("\n" + "=" * 60)
print("EXEMPLOS DE RELEVÂNCIA")
for i in range(min(10, len(relevancias))):
    print(relevancias[i])


print(30 * '=-')

print(relevancias.info)    

print(30 * '-=', 'Análises profundas abaixo', '-=' * 30 )


print("\n" + "=" * 80)
print("PRIMEIRO DOCUMENTO")
print("=" * 80)

doc = documentos[0]

for chave, valor in doc.items():
    print(f"{chave}: {valor}")

print("\n" + "=" * 80)
print("PRIMEIRAS 10 CONSULTAS")
print("=" * 80)

for i in range(10):
    print(consultas[i])

print("\n" + "=" * 80)
print("PRIMEIRAS 20 RELEVÂNCIAS")
print("=" * 80)

for i in range(20):
    print(relevancias[i])


print("\n" + "=" * 80)
print("VALORES ÚNICOS DE SOURCE")

print(relevancias.info)    

print(30 * '-=', 'Análises profundas abaixo', '-=' * 30 )


print("\n" + "=" * 80)
print("PRIMEIRO DOCUMENTO")
print("=" * 80)

doc = documentos[0]

for chave, valor in doc.items():
    print(f"{chave}: {valor}")

print("\n" + "=" * 80)
print("PRIMEIRAS 10 CONSULTAS")
print("=" * 80)

for i in range(10):
    print(consultas[i])

print("\n" + "=" * 80)
print("PRIMEIRAS 20 RELEVÂNCIAS")
print("=" * 80)

for i in range(20):
    print(relevancias[i])


print("\n" + "=" * 80)
print("VALORES ÚNICOS DE SOURCE")
print("=" * 80)

print(set(consultas["SOURCE"]))

print("\n" + "=" * 80)
print("VALORES ÚNICOS DE SCORE")
print("=" * 80)

print(sorted(set(relevancias["SCORE"])))

print("\n" + "=" * 80)
print("VALORES ÚNICOS DE ENGINE")
print("=" * 80)

print(set(relevancias["ENGINE"]))


print("\n" + "=" * 80)
print("EXEMPLO COMPLETO: CONSULTA + QRELS")
print("=" * 80)

query = consultas[0]

print("Consulta:")
print(query)

query_id = query["ID"]

qrels_query = relevancias.filter(
    lambda x: x["QUERY_ID"] == query_id
)

print("\nDocumentos avaliados para essa consulta:")

for qrel in qrels_query:
    print(qrel)

print(50 * '-=', 'Análises finais do dataset', '-=' * 50 )

print("\n" + "=" * 80)
print("VERIFICANDO RELAÇÃO DOC_ID -> DOCUMENTO")
print("=" * 80)

keys = set(documentos["KEY"])

doc_ids = set(relevancias["DOC_ID"])

encontrados = doc_ids.intersection(keys)

print(f"Qrels únicos DOC_ID: {len(doc_ids)}")
print(f"DOC_ID encontrados em KEY: {len(encontrados)}")
print(f"DOC_ID não encontrados em KEY: {len(doc_ids - keys)}")

print("\nExemplos encontrados:")
print(list(encontrados)[:20])

print("\nExemplos não encontrados:")
print(list(doc_ids - keys)[:20])

primeiro_doc_id = relevancias[0]["DOC_ID"]

print("\nProcurando primeiro DOC_ID:", primeiro_doc_id)

resultado = documentos.filter(
    lambda x: x["KEY"] == primeiro_doc_id
)

print(resultado)

if len(resultado) > 0:
    print(resultado[0])

from collections import Counter

print("\n" + "=" * 80)
print("DISTRIBUIÇÃO DAS CONSULTAS")
print("=" * 80)

distribuicao = Counter(consultas["SOURCE"])

for source, quantidade in distribuicao.items():
    print(f"{source}: {quantidade}")    



    