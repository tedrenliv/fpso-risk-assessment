import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Skill:
    domain_id: str
    domain_name: str
    version: str
    hazard_keywords: List[str]
    linguistic_variables: Dict[str, Dict[str, List[float]]]
    inference_templates: List[Dict[str, Any]]
    routing_confidence: float
    benchmark: Dict[str, Any]

    def get_tfn_terms(self, variable: str):
        """Return TFN objects for a linguistic variable."""
        from core.fuzzy_engine import TFN
        raw = self.linguistic_variables.get(variable, {})
        return {term: TFN(*vals) for term, vals in raw.items()}


class SkillRepository:
    SKILL_FILES = {
        "SI":   "structural_integrity.json",
        "MM":   "maintenance_management.json",
        "EH":   "environmental_hazard.json",
        "HF":   "human_factors.json",
        "SysI": "system_integration.json",
    }

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir

    def load(self, domain_id: str) -> Skill:
        filename = self.SKILL_FILES[domain_id]
        path = os.path.join(self.skills_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Skill(**data)

    def load_all(self) -> List[Skill]:
        return [self.load(did) for did in self.SKILL_FILES]

    def save(self, skill: Skill, skills_dir: str = None) -> None:
        target_dir = skills_dir or self.skills_dir
        filename = self.SKILL_FILES[skill.domain_id]
        path = os.path.join(target_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill.__dict__, f, indent=2)
