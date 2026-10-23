from pathlib import Path

from data.provenance import environment_manifest


def test_nested_installed_directory_does_not_claim_ancestor_checkout(monkeypatch, tmp_path):
    installed = tmp_path/'site-packages'
    installed.mkdir()
    commands = []
    import subprocess
    actual = subprocess.check_output
    def git(command, **kwargs):
        if isinstance(command, str) or Path(command[0]).stem.lower() != 'git':
            return actual(command, **kwargs)
        commands.append(command)
        assert command[-1] == '--show-toplevel', 'Ancestor checkout files must never be traversed.'
        return str(tmp_path)
    monkeypatch.setattr('data.provenance.subprocess.check_output', git)
    result = environment_manifest(installed)
    assert result['git_commit'] is None
    assert result['working_tree_code_sha256'] is None
    assert len(commands) == 1


def test_actual_checkout_source_identity_is_distinct_from_installed_identity():
    result = environment_manifest(Path(__file__).resolve().parents[1])
    assert len(result['git_commit']) == 40
    assert len(result['working_tree_code_sha256']) == 64
    assert 'installed_project_code_sha256' in result
