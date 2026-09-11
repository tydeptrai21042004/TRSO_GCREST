from tools.extended_paper_protocol import SESSIONS, session_payload


def test_revision_sessions_are_separate_from_canonical_protocol():
    assert set(SESSIONS) == set(range(7, 22))
    for session in range(7, 22):
        payload = session_payload(session)
        assert payload["canonical_46_run_protocol"] is False
        assert payload["proposal_default_unchanged"] is True
        assert payload["main_revision_expected_runs"] == 204
        if session != 17:
            assert payload["commands"]


def test_main_revision_sessions_use_exact_paper_matrix_and_batches():
    assert "full,linear,prompt,conv,piggyback,trso" in SESSIONS[7].commands[0]
    assert "fact_tt,fact_tk,vqt,spt_lora,spt_adapter" in next(x for x in SESSIONS[8].commands[0] if isinstance(x, str) and "fact_tt" in x)
    assert "16" in SESSIONS[11].commands[0]
    assert "ml_decoder" in next(x for x in SESSIONS[15].commands[0] if isinstance(x, str) and "ml_decoder" in x)
    assert "segadapter" in next(x for x in SESSIONS[16].commands[0] if isinstance(x, str) and "segadapter" in x)
