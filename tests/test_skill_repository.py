import pytest
from core.skill_repository import SkillRepository, Skill

def test_load_all_five_skills():
    repo = SkillRepository("C:/Multiagent/skills")
    skills = repo.load_all()
    assert len(skills) == 5
    ids = {s.domain_id for s in skills}
    assert ids == {"SI", "MM", "EH", "HF", "SysI"}

def test_skill_has_required_fields():
    repo = SkillRepository("C:/Multiagent/skills")
    skill = repo.load("SI")
    assert skill.domain_id == "SI"
    assert skill.domain_name
    assert skill.linguistic_variables
    assert skill.inference_templates
    assert skill.benchmark

def test_skill_version_increments_on_save(tmp_path):
    repo = SkillRepository("C:/Multiagent/skills")
    skill = repo.load("SI")
    original_version = skill.benchmark["version"]
    skill.benchmark["version"] += 1
    repo.save(skill, str(tmp_path))
    reloaded = SkillRepository(str(tmp_path)).load("SI")
    assert reloaded.benchmark["version"] == original_version + 1

def test_get_inference_templates_for_skill():
    repo = SkillRepository("C:/Multiagent/skills")
    skill = repo.load("MM")
    assert len(skill.inference_templates) >= 1
    for t in skill.inference_templates:
        assert "id" in t
        assert "conditions" in t
        assert "output" in t
        assert "weight" in t
