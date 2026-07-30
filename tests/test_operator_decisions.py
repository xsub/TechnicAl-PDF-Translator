from translator.operator_decisions import build_selected_operator_decision, normalize_decision_text


def test_rendered_segment_without_action_is_not_saved() -> None:
    decision = build_selected_operator_decision(
        action="skip",
        edited_text="Migracja globalna: < 10 mg/dm²",
        original_text="Migracja globalna: < 10 mg/dm²",
    )

    assert decision is None


def test_changed_text_is_saved_as_edit_even_when_action_is_skip() -> None:
    decision = build_selected_operator_decision(
        action="skip",
        edited_text="Europejskie oświadczenie regulacyjne dotyczące farb SunPak FSP EcoPace",
        original_text="Europejska informacja regulacyjna dotycząca farb drukarskich SunPak FSP EcoPace",
    )

    assert decision == {
        "action": "edit",
        "text": "Europejskie oświadczenie regulacyjne dotyczące farb SunPak FSP EcoPace",
    }


def test_explicit_accept_is_saved_without_text_change() -> None:
    decision = build_selected_operator_decision(
        action="accept",
        edited_text="Migracja globalna: < 10 mg/dm²",
        original_text="Migracja globalna: < 10 mg/dm²",
    )

    assert decision == {
        "action": "accept",
        "text": "Migracja globalna: < 10 mg/dm²",
    }


def test_changed_text_wins_over_explicit_accept() -> None:
    decision = build_selected_operator_decision(
        action="accept",
        edited_text="wyschnięta powłoka farby",
        original_text="wysuszonej warstwie farby",
    )

    assert decision == {
        "action": "edit",
        "text": "wyschnięta powłoka farby",
    }


def test_phrase_refinement_note_selects_segment() -> None:
    decision = build_selected_operator_decision(
        action="skip",
        edited_text="Tekst o wyschniętej powłoce farby.",
        original_text="Tekst o wyschniętej powłoce farby.",
        note="Fragment poprawiony przez użytkownika: preferowana fraza `wyschnięta powłoka farby`.",
    )

    assert decision == {
        "action": "edit",
        "text": "Tekst o wyschniętej powłoce farby.",
        "note": "Fragment poprawiony przez użytkownika: preferowana fraza `wyschnięta powłoka farby`.",
    }


def test_whitespace_only_changes_do_not_select_segment() -> None:
    assert normalize_decision_text("  A\n\nB   C ") == "A B C"
    decision = build_selected_operator_decision(
        action="skip",
        edited_text="  A\n\nB   C ",
        original_text="A B C",
    )

    assert decision is None
