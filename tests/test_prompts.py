from app import prompts


def test_version_is_positive_int():
    assert isinstance(prompts.version(), int)
    assert prompts.version() >= 1


def test_qa_template_has_required_placeholders():
    template = prompts.template("qa")
    assert "{context}" in template
    assert "{question}" in template
    assert "{not_found}" in template


def test_faithfulness_judge_template_has_required_placeholders():
    template = prompts.template("faithfulness_judge")
    assert "{context}" in template
    assert "{answer}" in template


def test_not_found_message_is_nonempty():
    assert len(prompts.not_found_message()) > 10
