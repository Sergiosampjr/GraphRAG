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