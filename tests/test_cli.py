import pytest

from wiki_agents.orchestrator.cli import main as wiki_agent_main
from wiki_agents.orchestrator.stages.create_plans import main as create_plans_main
from wiki_agents.orchestrator.stages.extract_concepts import (
    main as extract_concepts_main,
)
from wiki_agents.orchestrator.stages.write_wiki import main as write_wiki_main


@pytest.mark.parametrize(
    "entry_point",
    [wiki_agent_main, create_plans_main, extract_concepts_main, write_wiki_main],
)
def test_user_facing_commands_offer_progress_flag(entry_point, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        entry_point(["--help"])

    assert exit_info.value.code == 0
    assert "--progress" in capsys.readouterr().out
