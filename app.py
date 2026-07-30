from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from translator.schemas import JobConfig
from translator.storage import JobStore
from translator.utils import new_job_id
from translator.workflow import finalize_with_operator_decisions, resume_mvp_pipeline, run_mvp_pipeline


load_dotenv(override=True)

STORAGE_INPUT = Path("storage/input")
STORAGE_OUTPUT = Path("storage/output")

PROVIDER_LABELS = {
    "openai": "OpenAI - realne tłumaczenie",
    "anthropic": "Anthropic - realny review",
    "mock": "Mock - tylko test przepływu",
}

MODE_LABELS = {
    "standard": "Standard",
    "high_assurance": "High assurance",
    "strict_regulatory": "Strict regulatory",
}


def _init_session_state() -> None:
    st.session_state.setdefault("last_error", None)
    if "translation_state" not in st.session_state:
        latest_state = _load_latest_checkpoint(silent=True)
        st.session_state["translation_state"] = latest_state
        st.session_state["loaded_checkpoint_job_id"] = latest_state.get("job_id") if latest_state else None
    else:
        st.session_state.setdefault("loaded_checkpoint_job_id", None)


def _load_latest_checkpoint(*, silent: bool = False) -> dict | None:
    try:
        return JobStore().load_latest_state()
    except Exception as exc:  # noqa: BLE001 - checkpoint load must not break app boot
        if not silent:
            st.session_state["last_error"] = f"Nie udało się wczytać checkpointu: {exc}"
        return None


def _has_env_value(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def _default_translator_provider() -> str:
    return "openai" if _has_env_value("OPENAI_API_KEY") else "mock"


def _default_reviewer_provider() -> str:
    if _has_env_value("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _has_env_value("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def _default_debug_enabled() -> bool:
    return os.getenv("TRANSLATOR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _provider_format(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


def _mode_format(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _validate_provider_configuration(translator_provider: str, reviewer_provider: str) -> list[str]:
    errors = []
    if translator_provider == "openai" and not _has_env_value("OPENAI_API_KEY"):
        errors.append("Brakuje `OPENAI_API_KEY` w `.env`, a wybrano tłumacza OpenAI.")
    if reviewer_provider == "openai" and not _has_env_value("OPENAI_API_KEY"):
        errors.append("Brakuje `OPENAI_API_KEY` w `.env`, a wybrano review OpenAI.")
    if reviewer_provider == "anthropic" and not _has_env_value("ANTHROPIC_API_KEY"):
        errors.append("Brakuje `ANTHROPIC_API_KEY` w `.env`, a wybrano review Anthropic.")
    return errors


def _issue_label(issue: object) -> str:
    severity = getattr(issue, "severity", "info")
    issue_type = getattr(issue, "issue_type", getattr(issue, "category", "issue"))
    message = getattr(issue, "message", getattr(issue, "explanation", ""))
    return f"{severity.upper()} - {issue_type}: {message}"


def _segment_issues(state: dict, segment_id: str) -> list[object]:
    issues: list[object] = [
        issue for issue in state.get("deterministic_issues", [])
        if issue.segment_id == segment_id
    ]
    for review in state.get("review_results", {}).values():
        issues.extend([finding for finding in review.findings if finding.segment_id == segment_id])
    return issues


def _issue_severity(issue: object) -> str:
    return str(getattr(issue, "severity", "info")).lower()


def _status_update(status, *, label: str, state: str | None = None) -> None:
    update_kwargs = {"label": label, "expanded": True}
    if state:
        update_kwargs["state"] = state
    status.update(**update_kwargs)


def _make_progress_callback(
    status,
    progress_bar,
    detail_slot,
    history_slot,
    live_preview_slot=None,
    *,
    checkpoint_job_id: str | None = None,
    checkpoint_source_path: str | None = None,
):
    last_stage = {"value": None}
    last_log_key = {"value": None}
    last_preview_key = {"value": None}

    def on_progress(event: dict) -> None:
        stage = str(event.get("stage") or "workflow")
        message = str(event.get("message") or "Pracuję...")
        current = event.get("current")
        total = event.get("total")
        segment_id = event.get("segment_id")

        if isinstance(current, int) and isinstance(total, int) and total > 0:
            ratio = min(max(current / total, 0.0), 1.0)
            progress_text = f"{message}: {current}/{total}"
            if segment_id:
                progress_text = f"{progress_text} - {segment_id}"
            progress_bar.progress(ratio, text=progress_text)
            detail_slot.caption(
                f"Etap: `{stage}` | segment: `{segment_id or '-'}` | postęp etapu: {current}/{total}"
            )
            log_key = (stage, current // 10)
        else:
            ratio = 1.0 if stage in {"done", "human_review"} else 0.0
            progress_bar.progress(ratio, text=message)
            detail_slot.caption(f"Etap: `{stage}`")
            log_key = (stage, message)

        _status_update(status, label=message)

        if stage != last_stage["value"]:
            history_slot.write(f"**{message}**")
            last_stage["value"] = stage
            last_log_key["value"] = log_key
        elif log_key != last_log_key["value"] and isinstance(current, int) and isinstance(total, int):
            history_slot.caption(f"{message}: {current}/{total}")
            last_log_key["value"] = log_key

        if live_preview_slot is not None and stage in {"extract", "prepare", "translate", "review", "revise"}:
            preview_state = _load_latest_checkpoint(silent=True)
            if not preview_state or not _checkpoint_matches(preview_state, checkpoint_job_id, checkpoint_source_path):
                return

            translations_count = len(preview_state.get("translations", {}))
            preview_key = (
                preview_state.get("job_id"),
                preview_state.get("status"),
                len(preview_state.get("segments", [])),
                translations_count,
                len(preview_state.get("review_results", {})),
            )
            if preview_key != last_preview_key["value"]:
                last_preview_key["value"] = preview_key
                with live_preview_slot.container():
                    _render_live_translation_preview(
                        preview_state,
                        title="Podgląd na żywo tłumaczenia",
                    )

    return on_progress


def _checkpoint_matches(
    state: dict,
    checkpoint_job_id: str | None,
    checkpoint_source_path: str | None,
) -> bool:
    if checkpoint_job_id:
        return state.get("job_id") == checkpoint_job_id
    if checkpoint_source_path:
        return str(state.get("source_pdf_path")) == str(checkpoint_source_path)
    return True


def _run_translation(
    uploaded_file,
    target_language: str,
    mode: str,
    translator_provider: str,
    reviewer_provider: str,
    require_human_review: bool,
    debug: bool,
) -> None:
    errors = _validate_provider_configuration(translator_provider, reviewer_provider)
    if errors:
        st.session_state["last_error"] = "\n".join(errors)
        return

    STORAGE_INPUT.mkdir(parents=True, exist_ok=True)
    input_path = STORAGE_INPUT / uploaded_file.name
    input_path.write_bytes(uploaded_file.getbuffer())
    job_id = new_job_id()

    config = JobConfig(
        source_pdf_path=str(input_path),
        output_dir=str(STORAGE_OUTPUT),
        target_language=target_language,
        mode=mode,  # type: ignore[arg-type]
        translator_provider=translator_provider,  # type: ignore[arg-type]
        reviewer_provider=reviewer_provider,  # type: ignore[arg-type]
        require_human_review=require_human_review,
        debug=debug,
        job_id=job_id,
    )

    status = st.status("Przetwarzam dokument...", expanded=True)
    status.write("Ekstrakcja PDF, segmentacja, tłumaczenie, walidacja i review.")
    if debug:
        status.caption("Tryb debug jest włączony - szczegółowe logi lecą do terminala i `storage/logs/`.")
    progress_bar = status.progress(0, text="Startuję workflow...")
    detail_slot = status.empty()
    history_slot = status.container(height=180, border=True)
    preview_slot = st.empty()
    with preview_slot.container():
        _render_live_translation_preview(
            {
                "job_id": job_id,
                "status": "created",
                "segments": [],
                "translations": {},
            },
            title="Podgląd na żywo tłumaczenia",
        )
    progress_callback = _make_progress_callback(
        status,
        progress_bar,
        detail_slot,
        history_slot,
        preview_slot,
        checkpoint_job_id=job_id,
    )
    try:
        state = run_mvp_pipeline(config, progress_callback=progress_callback)
    except Exception as exc:  # noqa: BLE001 - show user-facing failure in Streamlit
        latest_state = _load_latest_checkpoint(silent=True)
        if latest_state:
            st.session_state["translation_state"] = latest_state
            st.session_state["loaded_checkpoint_job_id"] = latest_state.get("job_id")
        st.session_state["last_error"] = str(exc)
        _status_update(status, label="Przetwarzanie przerwane", state="error")
        return

    st.session_state["translation_state"] = state
    st.session_state["loaded_checkpoint_job_id"] = None
    st.session_state["last_error"] = None
    if state.get("status") == "needs_human_review":
        _status_update(status, label="Wymagana decyzja operatora", state="complete")
    else:
        _status_update(status, label="Dokument przetworzony", state="complete")


def _render_status(state: dict) -> None:
    config = state.get("config")
    provider_note = ""
    if config:
        provider_note = f" | tłumacz: `{config.translator_provider}`, review: `{config.reviewer_provider}`"

    st.subheader("Status")
    st.write(f"`{state.get('status', 'unknown')}`{provider_note}")
    if config and config.debug:
        log_path = Path(config.output_dir).parent / "logs" / f"{state.get('job_id')}.debug.log"
        st.caption(f"Debug log: `{log_path}`")

    segments = state.get("segments", [])
    review_findings = sum(len(result.findings) for result in state.get("review_results", {}).values())
    unresolved_segments = state.get("unresolved_segments", [])

    cols = st.columns(5)
    cols[0].metric("Segmenty", len(segments))
    cols[1].metric("Walidacja", len(state.get("deterministic_issues", [])))
    cols[2].metric("Review", review_findings)
    cols[3].metric("Do decyzji", len(unresolved_segments))
    cols[4].metric("PDF", "tak" if state.get("output_pdf_path") else "nie")

    if config and config.translator_provider == "mock":
        st.warning(
            "Ten wynik powstał w trybie mock. To tryb testowy przepływu, nie prawdziwe tłumaczenie dokumentu. "
            "Dla realnego PDF-a wybierz tłumacza OpenAI."
        )


def _render_live_translation_preview(state: dict, *, title: str) -> None:
    segments = sorted(state.get("segments", []), key=lambda segment: segment.order_index)
    translations = state.get("translations", {})
    translated_pairs = [
        (segment, translations[segment.segment_id])
        for segment in segments
        if segment.segment_id in translations
    ]

    total = len(segments)
    done = len(translated_pairs)
    progress_ratio = done / total if total else 0.0

    with st.container(border=True):
        st.markdown(f"#### :material/visibility: {title}")
        st.progress(
            min(max(progress_ratio, 0.0), 1.0),
            text=f"Przetłumaczone i zapisane: {done}/{total or '?'}",
        )
        st.caption(
            "To jest odczyt z checkpointu SQLite. Po każdym segmencie panel odświeża się tym, "
            "co naprawdę zostało zapisane."
        )

        if not total:
            st.info("Ekstrahuję PDF i buduję segmenty. Za chwilę pojawi się lista tłumaczeń.")
            return

        if not translated_pairs:
            st.info("Segmenty są już przygotowane. Czekam na pierwszy zapisany segment tłumaczenia.")
            return

        last_segment, last_translation = translated_pairs[-1]
        st.markdown("Ostatnio zapisany segment")
        left, right = st.columns(2)
        left.caption(f"Źródło — {last_segment.segment_id}, strona {last_segment.page_number}")
        left.code(last_segment.source_text)
        right.caption("Tłumaczenie zapisane w checkpoincie")
        right.write(last_translation.translated_text)

        rows = [
            {
                "#": index,
                "segment": segment.segment_id,
                "strona": segment.page_number,
                "źródło": _short_text(segment.source_text, 240),
                "tłumaczenie": _short_text(translation.translated_text, 520),
                "pewność": translation.confidence,
            }
            for index, (segment, translation) in enumerate(translated_pairs, start=1)
        ]
        st.markdown("Wszystkie zapisane tłumaczenia w tej chwili")
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            height=min(520, 140 + 32 * min(len(rows), 12)),
        )

        full_text = _joined_translation_text(translated_pairs)
        st.text_area(
            "Sklejony tekst ze wszystkich dotąd przetłumaczonych segmentów",
            value=full_text,
            height=260,
            disabled=True,
            key=f"live_joined_translation_{state.get('job_id')}_{done}_{_text_digest(full_text)}",
        )


def _render_checkpoint_resume(state: dict) -> None:
    if not _can_resume(state):
        return

    translations = state.get("translations", {})
    segments = state.get("segments", [])
    st.warning(
        "Mam zapisany checkpoint tego joba. Jeśli Streamlit się przeładował albo przerwał pracę, "
        f"możesz kontynuować bez tracenia zapisanych segmentów ({len(translations)}/{len(segments)})."
    )
    if st.button("Kontynuuj ostatni job z checkpointu", type="primary", icon=":material/play_arrow:"):
        _resume_translation(state)
        st.rerun()


def _can_resume(state: dict) -> bool:
    if state.get("output_pdf_path"):
        return False
    if state.get("status") == "needs_human_review":
        return False

    segments = state.get("segments", [])
    translations = state.get("translations", {})
    review_results = state.get("review_results", {})

    if not segments:
        return True
    if len(translations) < len(segments):
        return True
    if len(review_results) < len(translations):
        return True

    return state.get("status") not in {"completed", "completed_with_output_warnings"}


def _resume_translation(state: dict) -> None:
    config = state.get("config")
    if not config:
        st.session_state["last_error"] = "Checkpoint nie zawiera konfiguracji joba."
        return

    source_path = Path(state.get("source_pdf_path", ""))
    if not source_path.exists():
        st.session_state["last_error"] = f"Nie mogę wznowić: brakuje pliku źródłowego `{source_path}`."
        return

    errors = _validate_provider_configuration(config.translator_provider, config.reviewer_provider)
    if errors:
        st.session_state["last_error"] = "\n".join(errors)
        return

    status = st.status("Kontynuuję z checkpointu...", expanded=True)
    progress_bar = status.progress(0, text="Wznawiam workflow...")
    detail_slot = status.empty()
    history_slot = status.container(height=180, border=True)
    preview_slot = st.empty()
    _render_translation_preview(
        state,
        title="Checkpoint przed wznowieniem",
        max_rows=8,
        compact=True,
    )
    progress_callback = _make_progress_callback(
        status,
        progress_bar,
        detail_slot,
        history_slot,
        preview_slot,
        checkpoint_job_id=state.get("job_id"),
    )
    try:
        resumed_state = resume_mvp_pipeline(state, progress_callback=progress_callback)
    except Exception as exc:  # noqa: BLE001 - keep visible partial result
        latest_state = JobStore().load_state(state["job_id"]) or _load_latest_checkpoint(silent=True)
        if latest_state:
            st.session_state["translation_state"] = latest_state
            st.session_state["loaded_checkpoint_job_id"] = latest_state.get("job_id")
        st.session_state["last_error"] = str(exc)
        _status_update(status, label="Wznowienie przerwane", state="error")
        return

    st.session_state["translation_state"] = resumed_state
    st.session_state["loaded_checkpoint_job_id"] = None
    st.session_state["last_error"] = None
    if resumed_state.get("status") == "needs_human_review":
        _status_update(status, label="Wymagana decyzja operatora", state="complete")
    else:
        _status_update(status, label="Workflow wznowiony i przetworzony", state="complete")


def _render_translation_preview(
    state: dict,
    *,
    title: str = "Podgląd tłumaczenia",
    max_rows: int | None = None,
    compact: bool = False,
) -> None:
    segments = sorted(state.get("segments", []), key=lambda segment: segment.order_index)
    translations = state.get("translations", {})
    translated_pairs = [
        (segment, translations[segment.segment_id])
        for segment in segments
        if segment.segment_id in translations
    ]

    if title:
        st.subheader(title)

    if not segments:
        st.info("Checkpoint nie zawiera jeszcze segmentów z PDF-a.")
        return

    if not translated_pairs:
        st.info(
            f"PDF został rozbity na {len(segments)} segmentów, ale nie ma jeszcze zapisanego tłumaczenia segmentu."
        )
        return

    st.caption(
        f"Zapisane tłumaczenia: {len(translated_pairs)}/{len(segments)}. "
        "Ten podgląd pochodzi z lokalnego checkpointu, więc przetrwa przeładowanie strony."
    )

    visible_pairs = translated_pairs if max_rows is None else translated_pairs[-max_rows:]
    rows = []
    for segment, translation in visible_pairs:
        rows.append(
            {
                "segment": segment.segment_id,
                "strona": segment.page_number,
                "typ": segment.block_type,
                "źródło": _short_text(segment.source_text, 220 if compact else 500),
                "tłumaczenie": _short_text(translation.translated_text, 320 if compact else 900),
                "pewność": translation.confidence,
            }
        )

    st.dataframe(rows, hide_index=True, width="stretch")
    if not compact:
        full_text = _joined_translation_text(translated_pairs)
        st.text_area(
            "Sklejony tekst ze wszystkich zapisanych tłumaczeń",
            value=full_text,
            height=320,
            disabled=True,
            key=f"checkpoint_joined_translation_{state.get('job_id')}_{len(translated_pairs)}_{_text_digest(full_text)}",
        )


def _short_text(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}…"


def _joined_translation_text(translated_pairs: list[tuple[object, object]]) -> str:
    parts = []
    for segment, translation in translated_pairs:
        page_number = getattr(segment, "page_number", "?")
        segment_id = getattr(segment, "segment_id", "?")
        translated_text = getattr(translation, "translated_text", "")
        parts.append(f"[strona {page_number} | {segment_id}]\n{translated_text}")
    return "\n\n".join(parts)


def _text_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _render_human_review(state: dict) -> None:
    unresolved_segments = state.get("unresolved_segments", [])
    if not unresolved_segments:
        return

    segments = {segment.segment_id: segment for segment in state.get("segments", [])}
    translations = state.get("translations", {})

    st.subheader("Decyzje operatora")
    with st.container(border=True):
        st.write(
            "Nie renderuję wszystkich problemów naraz, bo przy dużym PDF-ie UI staje się nieużywalny. "
            "Możesz przejrzeć próbkę albo świadomie zaakceptować aktualne tłumaczenia zbiorczo."
        )
        st.warning(
            "Akcja zbiorcza ma sens tylko po realnym tłumaczeniu. Po trybie mock wygeneruje PDF z wieloma fragmentami po angielsku."
        )

        with st.container(horizontal=True):
            if st.button(
                "Zaakceptuj wszystkie aktualne i wygeneruj PDF",
                type="secondary",
                icon=":material/done_all:",
            ):
                decisions = {segment_id: {"action": "accept"} for segment_id in unresolved_segments}
                status = st.status("Generuję PDF po zbiorczej akceptacji...", expanded=True)
                progress_bar = status.progress(0, text="Startuję renderowanie...")
                detail_slot = status.empty()
                history_slot = status.container(height=140, border=True)
                progress_callback = _make_progress_callback(status, progress_bar, detail_slot, history_slot)
                final_state = finalize_with_operator_decisions(state, decisions, progress_callback=progress_callback)
                _status_update(status, label="PDF wygenerowany po zbiorczej akceptacji", state="complete")
                st.session_state["translation_state"] = final_state
                st.session_state["loaded_checkpoint_job_id"] = None
                st.rerun()

            if st.button("Ukryj wynik w tej sesji i zacznij od nowa", icon=":material/refresh:"):
                st.session_state["translation_state"] = None
                st.session_state["loaded_checkpoint_job_id"] = None
                st.rerun()

    all_severities = sorted(
        {
            _issue_severity(issue)
            for segment_id in unresolved_segments
            for issue in _segment_issues(state, segment_id)
        }
    )
    if not all_severities:
        all_severities = ["manual_review"]

    default_severities = [severity for severity in ["critical", "major", "warning"] if severity in all_severities]
    selected_severities = st.pills(
        "Filtr problemów",
        all_severities,
        default=default_severities or all_severities,
        selection_mode="multi",
        key="issue_severity_filter",
    )
    if not selected_severities:
        selected_severities = all_severities

    filtered_segments = [
        segment_id for segment_id in unresolved_segments
        if not _segment_issues(state, segment_id)
        or any(_issue_severity(issue) in selected_severities for issue in _segment_issues(state, segment_id))
    ]
    if not filtered_segments:
        st.info("Brak segmentów dla wybranego filtra.")
        return

    max_to_show = st.slider(
        "Ile segmentów pokazać",
        min_value=1,
        max_value=min(50, len(filtered_segments)),
        value=min(10, len(filtered_segments)),
        key="review_limit",
    )

    decisions = {}
    for index, segment_id in enumerate(filtered_segments[:max_to_show]):
        segment = segments[segment_id]
        translation = translations[segment_id]
        issues = _segment_issues(state, segment_id)

        with st.expander(f"{segment_id} - strona {segment.page_number}", expanded=index == 0):
            left, right = st.columns(2)
            left.markdown("Źródło")
            left.code(segment.source_text)
            right.markdown("Tłumaczenie")
            edited = right.text_area(
                "Zatwierdzony tekst",
                value=translation.translated_text,
                key=f"edit_{segment_id}",
            )

            action = st.segmented_control(
                "Decyzja",
                ["edit", "accept", "keep_source"],
                default="edit",
                format_func={
                    "edit": "Zapisz edycję",
                    "accept": "Akceptuj",
                    "keep_source": "Zostaw źródło",
                }.get,
                key=f"action_{segment_id}",
            )

            if issues:
                st.markdown("Problemy")
                for issue in issues:
                    st.warning(_issue_label(issue))

            decisions[segment_id] = {"action": action or "edit", "text": edited}

    with st.container(horizontal=True):
        if st.button("Zapisz decyzje dla pokazanych segmentów", type="primary", icon=":material/save:"):
            status = st.status("Zapisuję decyzje i generuję wynik...", expanded=True)
            progress_bar = status.progress(0, text="Startuję renderowanie...")
            detail_slot = status.empty()
            history_slot = status.container(height=140, border=True)
            progress_callback = _make_progress_callback(status, progress_bar, detail_slot, history_slot)
            final_state = finalize_with_operator_decisions(state, decisions, progress_callback=progress_callback)
            _status_update(status, label="Decyzje zapisane i wynik wygenerowany", state="complete")
            st.session_state["translation_state"] = final_state
            st.session_state["loaded_checkpoint_job_id"] = None
            st.rerun()

        st.caption(f"Pokazuję {min(max_to_show, len(filtered_segments))} z {len(filtered_segments)} segmentów po filtrze.")


def _render_outputs(state: dict) -> None:
    if not state.get("output_pdf_path"):
        return

    output_path = Path(state["output_pdf_path"])
    report_path = Path(state["report_path"]) if state.get("report_path") else None
    st.success("PDF wynikowy jest gotowy.")
    with st.container(horizontal=True):
        st.download_button(
            "Pobierz PDF",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
        )
        if report_path and report_path.exists():
            st.download_button(
                "Pobierz raport JSON",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="application/json",
                icon=":material/download:",
            )


st.set_page_config(page_title="Technical PDF Translator", layout="wide")
_init_session_state()

st.title("Technical PDF Translator MVP")
st.caption("PDF -> segmenty -> tłumaczenie -> walidacja -> review -> decyzje operatora -> PDF.")

openai_ready = _has_env_value("OPENAI_API_KEY")
anthropic_ready = _has_env_value("ANTHROPIC_API_KEY")

with st.sidebar:
    st.header("Konfiguracja")
    st.caption(f"OpenAI: {'gotowy' if openai_ready else 'brak OPENAI_API_KEY'}")
    st.caption(f"Anthropic: {'gotowy' if anthropic_ready else 'brak ANTHROPIC_API_KEY'}")

    target_language = st.selectbox("Język docelowy", ["Polish"], index=0, key="target_language")
    mode = st.segmented_control(
        "Tryb",
        ["standard", "high_assurance", "strict_regulatory"],
        default="standard",
        format_func=_mode_format,
        key="mode",
        width="stretch",
    )
    translator_provider = st.segmented_control(
        "Tłumacz",
        ["openai", "mock"],
        default=_default_translator_provider(),
        format_func=_provider_format,
        key="translator_provider",
        width="stretch",
    )
    reviewer_provider = st.segmented_control(
        "Recenzent",
        ["anthropic", "openai", "mock"],
        default=_default_reviewer_provider(),
        format_func=_provider_format,
        key="reviewer_provider",
        width="stretch",
    )
    require_human_review = st.toggle("Zatrzymaj przy istotnych problemach", value=True, key="require_human_review")
    debug = st.toggle(
        "Tryb debug - logi w terminalu",
        value=_default_debug_enabled(),
        key="debug",
        help="Wypisuje do konsoli każdy etap, segment, request LLM, czas trwania i błędy. Nie loguje kluczy API.",
    )

if translator_provider == "mock":
    st.warning("Wybrany jest mock. To sprawdza przepływ aplikacji, ale nie tłumaczy prawdziwego dokumentu.")

with st.form("translation_form"):
    uploaded = st.file_uploader("Wybierz cyfrowy PDF", type=["pdf"])
    submitted = st.form_submit_button("Rozpocznij tłumaczenie", type="primary", icon=":material/translate:")

if submitted:
    if uploaded is None:
        st.error("Najpierw wybierz PDF.")
    else:
        _run_translation(
            uploaded,
            target_language,
            mode or "standard",
            translator_provider or "mock",
            reviewer_provider or "mock",
            require_human_review,
            debug,
        )

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

state = st.session_state.get("translation_state")
if state:
    if st.session_state.get("loaded_checkpoint_job_id") == state.get("job_id"):
        st.info(
            f"Wczytałem ostatni zapisany checkpoint: `{state.get('job_id')}` "
            f"ze statusem `{state.get('status', 'unknown')}`."
        )
    _render_status(state)
    _render_checkpoint_resume(state)
    _render_translation_preview(state)
    _render_human_review(state)
    _render_outputs(state)
else:
    st.info("Wrzuć PDF i uruchom workflow. Jeśli masz klucz w `.env`, domyślnie użyję OpenAI zamiast mocka.")
