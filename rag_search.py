import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

print("All liabraries loaded!")

df=pd.read_csv("match_data.csv")
df= df[df["Winner"]!="No Result"]
print("Matches loaded:",len(df))

sentences=[]
for index,row in df.iterrows():
    sentence=f"{row['Team1']} vs {row['Team2']} at {row['Venue']} on {row['Date']}.{row['Winner']} won."
    sentences.append(sentence)

print("\nMatch Sentences:")
for s in sentences:
    print(s)

model=SentenceTransformer("all-MiniLM-L6-v2")
embed=model.encode(sentences)
print("Embedding Shape:",embed.shape)
print("Each sentence is now a vector of",embed.shape[1],"numbers")

client=chromadb.Client()
collection =client.create_collection("cricket_matches")
collection.add(
    documents=sentences,
    embeddings=embed.tolist(),ids=[str(i) for i in range (len(sentences))]
)

print("\n All matches stored in chromadb.\n Total matches stored:",collection.count())

def search_matches(query,top_k=3):
    print(f"\n searchinf for:'{query}'")

    query_vector=model.encode([query]).tolist()

    results=collection.query(
        query_embeddings=query_vector,n_results=top_k)
    
    for i,doc in enumerate(results["documents"][0]):
        print(f"{i+1}.{doc}")

search_matches("England won match")