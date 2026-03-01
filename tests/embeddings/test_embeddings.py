import sys
import numpy as np

sys.path.insert(0, '/content')

from core.embeddings import EmbeddingPipeline
from core.vector_utils import top_k_similar, cosine_similarity, batch_cosine_similarity
from utils.helpers import (save_embeddings, load_embeddings,
                           save_embeddings_with_metadata,
                           load_embeddings_with_metadata,
                           inspect_embeddings)

pipeline = EmbeddingPipeline()
print(pipeline)


def test_english():
    chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks consist of layers of interconnected nodes.",
        "Karachi is the largest city in Pakistan by population.",
        "Water boils at 100 degrees Celsius at sea level.",
        "",
    ]
    embeddings = pipeline.embed_batch(chunks, is_query=False)
    assert embeddings.shape == (5, 768)

    single = pipeline.embed_single("Hello world", is_query=False)
    assert single.shape == (768,)
    print(" test_english passed")


def test_urdu():
    chunks = [
        "مشین لرننگ مصنوعی ذہانت کا ایک حصہ ہے",
        "نیورل نیٹ ورک میں کئی تہیں ہوتی ہیں",
        "کراچی پاکستان کا سب سے بڑا شہر ہے",
        "پانی سو ڈگری سینٹی گریڈ پر ابلتا ہے",
    ]
    embeddings = pipeline.embed_batch(chunks, is_query=False)
    assert embeddings.shape == (4, 768)
    print(" test_urdu passed")


def test_multilingual_similarity():
    eng = pipeline.embed_single("Machine learning is a subset of AI.", is_query=False)
    ur_same = pipeline.embed_single("مشین لرننگ مصنوعی ذہانت کا ایک حصہ ہے", is_query=False)
    ur_diff = pipeline.embed_single("کراچی پاکستان کا سب سے بڑا شہر ہے", is_query=False)

    same = cosine_similarity(eng, ur_same)
    diff = cosine_similarity(eng, ur_diff)

    print(f"  same meaning (EN vs UR AI)  : {same:.4f}")
    print(f"  diff topic  (EN AI vs city) : {diff:.4f}")
    assert same > diff
    print("test_multilingual_similarity passed")


def test_top_k_search():
    chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks consist of layers of interconnected nodes.",
        "Karachi is the largest city in Pakistan by population.",
        "Water boils at 100 degrees Celsius at sea level.",
        "مشین لرننگ مصنوعی ذہانت کا ایک حصہ ہے",
        "کراچی پاکستان کا سب سے بڑا شہر ہے",
    ]
    corpus = pipeline.embed_batch(chunks, is_query=False)

    query_emb = pipeline.embed_single("What is artificial intelligence?", is_query=True)
    results = top_k_similar(query_emb, corpus, k=3)
    print("  English query results:")
    for r in results:
        print(f"    [{r['index']}] {r['score']:.4f} → {chunks[r['index']]}")

    ur_query_emb = pipeline.embed_single("مصنوعی ذہانت کیا ہے", is_query=True)
    ur_results = top_k_similar(ur_query_emb, corpus, k=3)
    print("  Urdu query results:")
    for r in ur_results:
        print(f"    [{r['index']}] {r['score']:.4f} → {chunks[r['index']]}")

    print("test_top_k_search passed")


def test_save_load():
    chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks consist of layers of interconnected nodes.",
        "Karachi is the largest city in Pakistan by population.",
        "Water boils at 100 degrees Celsius at sea level.",
        "مشین لرننگ مصنوعی ذہانت کا ایک حصہ ہے",
        "کراچی پاکستان کا سب سے بڑا شہر ہے",
    ]
    corpus = pipeline.embed_batch(chunks, is_query=False)
    metadata = [{"text": c, "chunk_id": i} for i, c in enumerate(chunks)]

    save_embeddings_with_metadata(corpus, metadata,
                                  "/content/corpus_embeddings.npy",
                                  "/content/corpus_metadata.npy")

    loaded_emb, loaded_meta = load_embeddings_with_metadata(
        "/content/corpus_embeddings.npy",
        "/content/corpus_metadata.npy"
    )

    assert np.allclose(corpus, loaded_emb)
    assert len(loaded_meta) == len(chunks)

    query_emb = pipeline.embed_single("neural network layers", is_query=True)
    results = top_k_similar(query_emb, loaded_emb, k=2)
    print("  Search on loaded embeddings:")
    for r in results:
        print(f"    [{r['index']}] {r['score']:.4f} → {loaded_meta[r['index']]['text']}")

    inspect_embeddings(loaded_emb)
    print(" test_save_load passed")


def test_embed_documents():
    chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks consist of layers of interconnected nodes.",
        "Karachi is the largest city in Pakistan by population.",
    ]
    docs = [{"text": c, "source": "test.pdf", "chunk_id": i} for i, c in enumerate(chunks)]
    docs = pipeline.embed_documents(docs)

    assert "embedding" in docs[0]
    assert docs[0]["embedding"].shape == (768,)
    print("test_embed_documents passed")


if __name__ == "__main__":
    test_english()
    test_urdu()
    test_multilingual_similarity()
    test_top_k_search()
    test_save_load()
    test_embed_documents()
    print("\n✅ ALL TESTS PASSED")
