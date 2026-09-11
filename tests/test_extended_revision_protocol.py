from tools.extended_paper_protocol import SESSIONS, session_payload


def test_revision_sessions_are_separate_from_canonical_protocol():
    assert set(SESSIONS) == set(range(7, 15))
    for session in range(7, 15):
        payload = session_payload(session)
        assert payload["canonical_46_run_protocol"] is False
        assert payload["proposal_default_unchanged"] is True
        assert payload["commands"]
