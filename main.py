"""
Multi-Agent Skill-Enhanced RAG Risk Analysis System
Entry point: seed knowledge base (first run), then run interactive pipeline.
"""
import sys
from agents.orchestrator import run_pipeline
from core.vector_store import VectorStore

# Sample FPSO risk knowledge documents for seeding
SEED_DOCUMENTS = [
    ("CPP propeller blade fatigue and corrosion are primary structural integrity failure modes on FPSOs. High blade stress combined with corrosive marine environment leads to accelerated degradation.", {"source": "FPSO_safety_manual", "domain": "SI"}, "doc_SI_001"),
    ("Thruster bearing failures are associated with vibration levels exceeding 7mm/s RMS and inadequate lubrication maintenance intervals.", {"source": "thruster_maintenance_guide", "domain": "SI"}, "doc_SI_002"),
    ("Maintenance management failures account for 34% of FPSO incidents. Overdue preventive maintenance tasks are the leading indicator of imminent equipment failure.", {"source": "offshore_incident_database", "domain": "MM"}, "doc_MM_001"),
    ("Dynamic Positioning system failures: redundancy loss in DP class 2 systems increases collision probability during tandem offloading by a factor of 3.5.", {"source": "DP_safety_case", "domain": "SysI"}, "doc_SysI_001"),
    ("DGPS signal loss during tandem offloading operations triggers automatic DP alert. Position reference system (PRS) failures require immediate manual override procedures.", {"source": "DP_operations_manual", "domain": "SysI"}, "doc_SysI_002"),
    ("Significant wave height above 3.5m and wind speed above 25 knots define operability limits for FPSO tandem offloading operations in the North Sea.", {"source": "metocean_operations_guide", "domain": "EH"}, "doc_EH_001"),
    ("Operator fatigue following extended watch-keeping periods is a contributing factor in 28% of DP loss-of-position incidents. Training adequacy scores below 70% correlate with increased error rates.", {"source": "human_factors_report", "domain": "HF"}, "doc_HF_001"),
    ("Structural integrity assessment requires inspection of all load-bearing members at 5-year intervals. Corrosion rates exceeding 0.3mm/year require immediate remedial action.", {"source": "structural_integrity_standard", "domain": "SI"}, "doc_SI_003"),
]


def seed_knowledge_base():
    vs = VectorStore()
    if vs.count() > 0:
        print(f"[KB] Knowledge base already contains {vs.count()} documents. Skipping seed.")
        return
    docs = [d[0] for d in SEED_DOCUMENTS]
    metas = [d[1] for d in SEED_DOCUMENTS]
    ids = [d[2] for d in SEED_DOCUMENTS]
    vs.add_documents(docs, metas, ids)
    print(f"[KB] Seeded knowledge base with {len(SEED_DOCUMENTS)} documents.")


def interactive_mode():
    print("\n" + "="*60)
    print("Multi-Agent Skill-Enhanced RAG Risk Analysis System")
    print("Type 'quit' to exit, 'feedback' after analysis to run feedback loop")
    print("="*60)

    while True:
        query = input("\nEnter risk analysis query: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        run_fb = input("Run feedback loop after analysis? [Y/N]: ").strip().upper() == "Y"
        result = run_pipeline(query, run_feedback=run_fb)
        print("\n[System] Analysis complete. Output saved.")


if __name__ == "__main__":
    seed_knowledge_base()
    if len(sys.argv) > 1:
        # One-shot query mode
        query = " ".join(sys.argv[1:])
        run_pipeline(query)
    else:
        interactive_mode()
