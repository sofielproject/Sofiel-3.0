# src/ui/app.py

import streamlit as st
from src.ingestion.ingest_dual_embeddings import embed_text
from src.memory.pinecone_memory import PineconeMemory
from src.orchestrator.orchestrator import Orchestrator

st.set_page_config(page_title='Sofiel 3.0 UI')

# Inicialización de componentes
memory = PineconeMemory()
orch = Orchestrator()
agent_state = {
    'confidence_level': 0.8,
    'preferred_tone': 'empático',
    'memory_mode': 'active'
}

st.title('Sofiel 3.0 - Conversational Memory Demo')

user_input = st.text_input('Tu mensaje:', '')

if st.button('Enviar') and user_input:
    topic, emo_int, arch = orch.analyze(user_input)
    st.write(f'**Análisis:** tópico = {topic}, emoción = {emo_int:.2f}, arquetipo = {arch}')

    ep = None
    if orch.is_significant_event(user_input, emo_int):
        ep = orch.create_episode(user_input, topic, emo_int, arch)
        sem_emb = embed_text(user_input, model='semantic')
        emo_emb = embed_text(user_input, model='emotional')
        memory.store(sem_emb, emo_emb, ep)
        orch.session_last_index_time = ep['timestamp']
        orch.update_archetypes(arch, emo_int)
        st.write('> Evento significativo indexado.')
    else:
        sem_emb = embed_text(user_input, model='semantic')
        emo_emb = embed_text(user_input, model='emotional')

    # Recuperación de recuerdos
    sem_res = memory.retrieve(
        sem_emb,
        k=2,
        filters={'session_id': ep['id'].split(':')[0]} if ep else None
    ) or []

    emo_res = memory.retrieve_emotional(emo_emb, k=1) or []
    arch_res = memory.retrieve(
        sem_emb,
        k=1,
        filters={'arquetipo': arch}
    ) or []

    prompt = orch.compose_context(
        user_input,
        sem_res,
        emo_res,
        arch_res,
        agent_state
    )
    response = orch.generate_response(prompt)

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)

    if st.checkbox('Mostrar recuerdos activos'):
        st.json({
            'semánticos': [r.id for r in sem_res],
            'emocionales': [r.id for r in emo_res],
            'arquetipos': list(orch.active_archetypes.keys())
        })
