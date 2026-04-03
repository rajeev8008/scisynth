# QASPER-style long-document QA

Long-document question answering requires chunk-level retrieval before synthesis.
Models need to gather evidence from multiple sections instead of relying on a
single passage. Evaluation often tracks answer relevance and faithfulness.

Section methods: retrieval quality strongly affects downstream generation.
Section findings: multi-hop retrieval can improve synthesis when evidence is
distributed across a paper.
