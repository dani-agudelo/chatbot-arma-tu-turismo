flowchart TD
    docs[Documentos PDF] --> chunker[Chunking]
    chunker --> embedDocs[ModeloEmbeddings]
    embedDocs --> chroma[(ChromaCollection)]

    userQ[PreguntaUsuario] --> embedQ[ModeloEmbeddings]
    embedQ --> retriever[BusquedaSimilitudTopK]
    chroma --> retriever

    retriever --> context[ChunksRecuperados]
    context --> llm[LLMGemini]
    userQ --> llm
    llm --> answer[RespuestaConFuentes]



Lectura del diagrama
- Embeddings se usan dos veces: para documentos y para pregunta.
- Con eso haces búsqueda vectorial en Chroma.
- El LLM no busca, el LLM redacta/razona con el contexto recuperado.
- Sin embeddings + vector DB, el LLM no tiene recuperación semántica eficiente sobre tu corpus.