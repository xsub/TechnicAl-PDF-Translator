from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from translator.languages import LANGUAGE_OPTIONS, language_index
from translator.schemas import JobConfig
from translator.storage import JobStore
from translator.utils import new_job_id
from translator.workflow import finalize_with_operator_decisions, resume_mvp_pipeline, run_mvp_pipeline


load_dotenv(override=True)

STORAGE_INPUT = Path("storage/input")
STORAGE_OUTPUT = Path("storage/output")

PROVIDER_LABELS = {
    "pl": {
        "openai": "OpenAI - realne tłumaczenie",
        "anthropic": "Anthropic - realny review",
        "mock": "Mock - tylko test przepływu",
    },
    "en": {
        "openai": "OpenAI - real translation",
        "anthropic": "Anthropic - real review",
        "mock": "Mock - workflow test only",
    },
}

MODE_LABELS = {
    "pl": {
        "standard": "Standard",
        "high_assurance": "High assurance",
        "strict_regulatory": "Strict regulatory",
    },
    "en": {
        "standard": "Standard",
        "high_assurance": "High assurance",
        "strict_regulatory": "Strict regulatory",
    },
}

UI_TEXT = {
    "pl": {
        "page_title": "Technical PDF Translator MVP",
        "page_caption": "PDF -> segmenty -> tłumaczenie -> walidacja -> review -> decyzje operatora -> PDF.",
        "configuration": "Konfiguracja",
        "openai_status": "OpenAI: {status}",
        "anthropic_status": "Anthropic: {status}",
        "ready": "gotowy",
        "missing_key": "brak {key}",
        "ui_language": "Język interfejsu / UI language",
        "translation_languages": "Języki tłumaczenia",
        "source_language": "Z języka",
        "target_language": "Na język",
        "language_placeholder": "Wybierz lub wpisz język",
        "language_help": "Lista jest przeszukiwalna. Możesz też wpisać własny język, np. Portuguese (Brazil) albo Serbian Latin.",
        "same_language_warning": "Wybrano ten sam język źródłowy i docelowy. To jest dozwolone, ale zwykle oznacza brak tłumaczenia.",
        "mode": "Tryb",
        "translator": "Tłumacz",
        "reviewer": "Recenzent",
        "require_human_review": "Zatrzymaj przy istotnych problemach",
        "debug": "Tryb debug - logi w terminalu",
        "debug_help": "Wypisuje do konsoli każdy etap, segment, request LLM, czas trwania i błędy. Nie loguje kluczy API.",
        "mock_warning": "Wybrany jest mock. To sprawdza przepływ aplikacji, ale nie tłumaczy prawdziwego dokumentu.",
        "upload_pdf": "Wybierz cyfrowy PDF",
        "start_translation": "Rozpocznij tłumaczenie",
        "select_pdf_first": "Najpierw wybierz PDF.",
        "empty_info": "Wrzuć PDF i uruchom workflow. Jeśli masz klucz w `.env`, domyślnie użyję OpenAI zamiast mocka.",
        "processing": "Przetwarzam dokument...",
        "processing_detail": "Ekstrakcja PDF, segmentacja, tłumaczenie, walidacja i review.",
        "debug_enabled": "Tryb debug jest włączony - szczegółowe logi lecą do terminala i `storage/logs/`.",
        "workflow_start": "Startuję workflow...",
        "interrupted": "Przetwarzanie przerwane",
        "needs_human_review": "Wymagana decyzja operatora",
        "processed": "Dokument przetworzony",
        "status": "Status",
        "translator_provider": "tłumacz",
        "review_provider": "review",
        "source_to_target": "języki",
        "debug_log": "Debug log: `{path}`",
        "segments_metric": "Segmenty",
        "validation_metric": "Walidacja",
        "review_metric": "Review",
        "decisions_metric": "Do decyzji",
        "pdf_metric": "PDF",
        "yes": "tak",
        "no": "nie",
        "mock_result_warning": "Ten wynik powstał w trybie mock. To tryb testowy przepływu, nie prawdziwe tłumaczenie dokumentu. Dla realnego PDF-a wybierz tłumacza OpenAI.",
        "live_preview": "Podgląd na żywo tłumaczenia",
        "translated_saved": "Przetłumaczone i zapisane: {done}/{total}",
        "checkpoint_caption": "To jest odczyt z checkpointu SQLite. Po każdym segmencie panel odświeża się tym, co naprawdę zostało zapisane.",
        "extracting_wait": "Ekstrahuję PDF i buduję segmenty. Za chwilę pojawi się lista tłumaczeń.",
        "waiting_first_translation": "Segmenty są już przygotowane. Czekam na pierwszy zapisany segment tłumaczenia.",
        "last_saved_segment": "Ostatnio zapisany segment",
        "source_segment_caption": "Źródło — {segment_id}, strona {page_number}",
        "saved_translation_caption": "Tłumaczenie zapisane w checkpoincie",
        "all_saved_translations": "Wszystkie zapisane tłumaczenia w tej chwili",
        "joined_live_text": "Sklejony tekst ze wszystkich dotąd przetłumaczonych segmentów",
        "resume_warning": "Mam zapisany checkpoint tego joba. Jeśli Streamlit się przeładował albo przerwał pracę, możesz kontynuować bez tracenia zapisanych segmentów ({done}/{total}).",
        "resume_button": "Kontynuuj ostatni job z checkpointu",
        "resume_status": "Kontynuuję z checkpointu...",
        "resume_progress": "Wznawiam workflow...",
        "checkpoint_before_resume": "Checkpoint przed wznowieniem",
        "resume_interrupted": "Wznowienie przerwane",
        "resume_done": "Workflow wznowiony i przetworzony",
        "translation_preview": "Podgląd tłumaczenia",
        "empty_checkpoint": "Checkpoint nie zawiera jeszcze segmentów z PDF-a.",
        "no_translation_yet": "PDF został rozbity na {total} segmentów, ale nie ma jeszcze zapisanego tłumaczenia segmentu.",
        "saved_translation_count": "Zapisane tłumaczenia: {done}/{total}. Ten podgląd pochodzi z lokalnego checkpointu, więc przetrwa przeładowanie strony.",
        "joined_saved_text": "Sklejony tekst ze wszystkich zapisanych tłumaczeń",
        "review_decisions": "Decyzje operatora",
        "bulk_review_note": "Nie renderuję wszystkich problemów naraz, bo przy dużym PDF-ie UI staje się nieużywalny. Możesz przejrzeć próbkę albo świadomie zaakceptować aktualne tłumaczenia zbiorczo.",
        "bulk_review_warning": "Akcja zbiorcza ma sens tylko po realnym tłumaczeniu. Po trybie mock wygeneruje PDF z wieloma fragmentami po angielsku.",
        "accept_all": "Zaakceptuj wszystkie aktualne i wygeneruj PDF",
        "rendering_after_accept": "Generuję PDF po zbiorczej akceptacji...",
        "rendering_start": "Startuję renderowanie...",
        "pdf_after_accept": "PDF wygenerowany po zbiorczej akceptacji",
        "hide_and_restart": "Ukryj wynik w tej sesji i zacznij od nowa",
        "issue_filter": "Filtr problemów",
        "no_segments_filter": "Brak segmentów dla wybranego filtra.",
        "segments_to_show": "Ile segmentów pokazać",
        "source": "Źródło",
        "translation": "Tłumaczenie",
        "approved_text": "Zatwierdzony tekst",
        "decision": "Decyzja",
        "action_edit": "Zapisz edycję",
        "action_accept": "Akceptuj",
        "action_keep_source": "Zostaw źródło",
        "issues": "Problemy",
        "save_decisions": "Zapisz decyzje dla pokazanych segmentów",
        "saving_decisions": "Zapisuję decyzje i generuję wynik...",
        "decisions_saved": "Decyzje zapisane i wynik wygenerowany",
        "showing_segments": "Pokazuję {shown} z {total} segmentów po filtrze.",
        "output_ready": "PDF wynikowy jest gotowy.",
        "download_pdf": "Pobierz PDF",
        "download_report": "Pobierz raport JSON",
        "loaded_checkpoint": "Wczytałem ostatni zapisany checkpoint: `{job_id}` ze statusem `{status}`.",
        "segment_col": "segment",
        "page_col": "strona",
        "type_col": "typ",
        "source_col": "źródło",
        "translation_col": "tłumaczenie",
        "confidence_col": "pewność",
        "missing_openai_translator": "Brakuje `OPENAI_API_KEY` w `.env`, a wybrano tłumacza OpenAI.",
        "missing_openai_reviewer": "Brakuje `OPENAI_API_KEY` w `.env`, a wybrano review OpenAI.",
        "missing_anthropic_reviewer": "Brakuje `ANTHROPIC_API_KEY` w `.env`, a wybrano review Anthropic.",
    },
    "en": {
        "page_title": "Technical PDF Translator MVP",
        "page_caption": "PDF -> segments -> translation -> validation -> review -> operator decisions -> PDF.",
        "configuration": "Configuration",
        "openai_status": "OpenAI: {status}",
        "anthropic_status": "Anthropic: {status}",
        "ready": "ready",
        "missing_key": "missing {key}",
        "ui_language": "UI language / język interfejsu",
        "translation_languages": "Translation languages",
        "source_language": "From language",
        "target_language": "To language",
        "language_placeholder": "Select or type a language",
        "language_help": "The list is searchable. You can also type a custom language, e.g. Portuguese (Brazil) or Serbian Latin.",
        "same_language_warning": "The source and target languages are the same. This is allowed, but usually means no translation will happen.",
        "mode": "Mode",
        "translator": "Translator",
        "reviewer": "Reviewer",
        "require_human_review": "Stop on material issues",
        "debug": "Debug mode - terminal logs",
        "debug_help": "Prints every stage, segment, LLM request, duration and error to the console. API keys are not logged.",
        "mock_warning": "Mock is selected. It tests the workflow, but does not really translate the document.",
        "upload_pdf": "Choose a digital PDF",
        "start_translation": "Start translation",
        "select_pdf_first": "Choose a PDF first.",
        "empty_info": "Upload a PDF and start the workflow. If `.env` contains an API key, OpenAI is used by default instead of mock.",
        "processing": "Processing document...",
        "processing_detail": "PDF extraction, segmentation, translation, validation and review.",
        "debug_enabled": "Debug mode is on - detailed logs go to the terminal and `storage/logs/`.",
        "workflow_start": "Starting workflow...",
        "interrupted": "Processing interrupted",
        "needs_human_review": "Operator decision required",
        "processed": "Document processed",
        "status": "Status",
        "translator_provider": "translator",
        "review_provider": "review",
        "source_to_target": "languages",
        "debug_log": "Debug log: `{path}`",
        "segments_metric": "Segments",
        "validation_metric": "Validation",
        "review_metric": "Review",
        "decisions_metric": "To decide",
        "pdf_metric": "PDF",
        "yes": "yes",
        "no": "no",
        "mock_result_warning": "This result was created in mock mode. It is a workflow test, not a real document translation. For a real PDF, choose the OpenAI translator.",
        "live_preview": "Live translation preview",
        "translated_saved": "Translated and saved: {done}/{total}",
        "checkpoint_caption": "This reads from the SQLite checkpoint. After each segment, the panel refreshes with what was actually saved.",
        "extracting_wait": "Extracting the PDF and building segments. Translations will appear here shortly.",
        "waiting_first_translation": "Segments are ready. Waiting for the first saved translation segment.",
        "last_saved_segment": "Last saved segment",
        "source_segment_caption": "Source — {segment_id}, page {page_number}",
        "saved_translation_caption": "Translation saved in checkpoint",
        "all_saved_translations": "All translations saved right now",
        "joined_live_text": "Combined text from all translated segments so far",
        "resume_warning": "This job has a saved checkpoint. If Streamlit reloaded or work was interrupted, you can continue without losing saved segments ({done}/{total}).",
        "resume_button": "Continue latest job from checkpoint",
        "resume_status": "Continuing from checkpoint...",
        "resume_progress": "Resuming workflow...",
        "checkpoint_before_resume": "Checkpoint before resume",
        "resume_interrupted": "Resume interrupted",
        "resume_done": "Workflow resumed and processed",
        "translation_preview": "Translation preview",
        "empty_checkpoint": "The checkpoint does not contain PDF segments yet.",
        "no_translation_yet": "The PDF has been split into {total} segments, but no segment translation has been saved yet.",
        "saved_translation_count": "Saved translations: {done}/{total}. This preview comes from the local checkpoint, so it survives page reloads.",
        "joined_saved_text": "Combined text from all saved translations",
        "review_decisions": "Operator decisions",
        "bulk_review_note": "I do not render all issues at once because large PDFs make the UI unwieldy. You can review a sample or intentionally accept current translations in bulk.",
        "bulk_review_warning": "Bulk acceptance makes sense only after real translation. After mock mode it will generate a PDF with many English fragments.",
        "accept_all": "Accept all current translations and generate PDF",
        "rendering_after_accept": "Generating PDF after bulk acceptance...",
        "rendering_start": "Starting rendering...",
        "pdf_after_accept": "PDF generated after bulk acceptance",
        "hide_and_restart": "Hide this result in this session and start again",
        "issue_filter": "Issue filter",
        "no_segments_filter": "No segments for the selected filter.",
        "segments_to_show": "How many segments to show",
        "source": "Source",
        "translation": "Translation",
        "approved_text": "Approved text",
        "decision": "Decision",
        "action_edit": "Save edit",
        "action_accept": "Accept",
        "action_keep_source": "Keep source",
        "issues": "Issues",
        "save_decisions": "Save decisions for shown segments",
        "saving_decisions": "Saving decisions and generating output...",
        "decisions_saved": "Decisions saved and output generated",
        "showing_segments": "Showing {shown} of {total} segments after filtering.",
        "output_ready": "Output PDF is ready.",
        "download_pdf": "Download PDF",
        "download_report": "Download JSON report",
        "loaded_checkpoint": "Loaded the latest checkpoint: `{job_id}` with status `{status}`.",
        "segment_col": "segment",
        "page_col": "page",
        "type_col": "type",
        "source_col": "source",
        "translation_col": "translation",
        "confidence_col": "confidence",
        "missing_openai_translator": "`OPENAI_API_KEY` is missing in `.env`, but OpenAI translator is selected.",
        "missing_openai_reviewer": "`OPENAI_API_KEY` is missing in `.env`, but OpenAI review is selected.",
        "missing_anthropic_reviewer": "`ANTHROPIC_API_KEY` is missing in `.env`, but Anthropic review is selected.",
    },
}


def _init_session_state() -> None:
    st.session_state.setdefault("last_error", None)
    if "translation_state" not in st.session_state:
        latest_state = _load_latest_checkpoint(silent=True)
        st.session_state["translation_state"] = latest_state
        st.session_state["loaded_checkpoint_job_id"] = latest_state.get("job_id") if latest_state else None
    else:
        st.session_state.setdefault("loaded_checkpoint_job_id", None)


def _ui_language() -> str:
    value = str(st.session_state.get("ui_language") or "pl").lower()
    return value if value in UI_TEXT else "pl"


def _t(key: str, **kwargs: object) -> str:
    template = UI_TEXT.get(_ui_language(), UI_TEXT["pl"]).get(key, UI_TEXT["pl"].get(key, key))
    return template.format(**kwargs) if kwargs else template


def _language_options_with_current(current: str | None) -> list[str]:
    options = list(LANGUAGE_OPTIONS)
    if current and current not in options:
        options.insert(0, current)
    return options


def _language_option_index(options: list[str], language: str) -> int:
    normalized = _normalized_language(language)
    for index, option in enumerate(options):
        if _normalized_language(option) == normalized:
            return index
    return language_index(language)


def _normalized_language(language: str | None) -> str:
    return " ".join((language or "").strip().lower().split())


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
    return PROVIDER_LABELS.get(_ui_language(), PROVIDER_LABELS["pl"]).get(provider, provider)


def _mode_format(mode: str) -> str:
    return MODE_LABELS.get(_ui_language(), MODE_LABELS["pl"]).get(mode, mode)


def _validate_provider_configuration(translator_provider: str, reviewer_provider: str) -> list[str]:
    errors = []
    if translator_provider == "openai" and not _has_env_value("OPENAI_API_KEY"):
        errors.append(_t("missing_openai_translator"))
    if reviewer_provider == "openai" and not _has_env_value("OPENAI_API_KEY"):
        errors.append(_t("missing_openai_reviewer"))
    if reviewer_provider == "anthropic" and not _has_env_value("ANTHROPIC_API_KEY"):
        errors.append(_t("missing_anthropic_reviewer"))
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
    source_language: str,
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
        source_language=source_language,
        target_language=target_language,
        mode=mode,  # type: ignore[arg-type]
        translator_provider=translator_provider,  # type: ignore[arg-type]
        reviewer_provider=reviewer_provider,  # type: ignore[arg-type]
        require_human_review=require_human_review,
        debug=debug,
        job_id=job_id,
    )

    status = st.status(_t("processing"), expanded=True)
    status.write(_t("processing_detail"))
    if debug:
        status.caption(_t("debug_enabled"))
    progress_bar = status.progress(0, text=_t("workflow_start"))
    detail_slot = status.empty()
    history_slot = status.container(height=180, border=True)
    preview_slot = st.empty()
    with preview_slot.container():
        _render_live_translation_preview(
            {
                "job_id": job_id,
                "status": "created",
                "source_language": source_language,
                "target_language": target_language,
                "segments": [],
                "translations": {},
            },
            title=_t("live_preview"),
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
        _status_update(status, label=_t("interrupted"), state="error")
        return

    st.session_state["translation_state"] = state
    st.session_state["loaded_checkpoint_job_id"] = None
    st.session_state["last_error"] = None
    if state.get("status") == "needs_human_review":
        _status_update(status, label=_t("needs_human_review"), state="complete")
    else:
        _status_update(status, label=_t("processed"), state="complete")


def _render_status(state: dict) -> None:
    config = state.get("config")
    provider_note = ""
    if config:
        provider_note = (
            f" | {_t('source_to_target')}: `{config.source_language} -> {config.target_language}`"
            f" | {_t('translator_provider')}: `{config.translator_provider}`, "
            f"{_t('review_provider')}: `{config.reviewer_provider}`"
        )

    st.subheader(_t("status"))
    st.write(f"`{state.get('status', 'unknown')}`{provider_note}")
    if config and config.debug:
        log_path = Path(config.output_dir).parent / "logs" / f"{state.get('job_id')}.debug.log"
        st.caption(_t("debug_log", path=log_path))

    segments = state.get("segments", [])
    review_findings = sum(len(result.findings) for result in state.get("review_results", {}).values())
    unresolved_segments = state.get("unresolved_segments", [])

    cols = st.columns(5)
    cols[0].metric(_t("segments_metric"), len(segments))
    cols[1].metric(_t("validation_metric"), len(state.get("deterministic_issues", [])))
    cols[2].metric(_t("review_metric"), review_findings)
    cols[3].metric(_t("decisions_metric"), len(unresolved_segments))
    cols[4].metric(_t("pdf_metric"), _t("yes") if state.get("output_pdf_path") else _t("no"))

    if config and config.translator_provider == "mock":
        st.warning(_t("mock_result_warning"))


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
            text=_t("translated_saved", done=done, total=total or "?"),
        )
        st.caption(_t("checkpoint_caption"))

        if not total:
            st.info(_t("extracting_wait"))
            return

        if not translated_pairs:
            st.info(_t("waiting_first_translation"))
            return

        last_segment, last_translation = translated_pairs[-1]
        st.markdown(_t("last_saved_segment"))
        left, right = st.columns(2)
        left.caption(_t("source_segment_caption", segment_id=last_segment.segment_id, page_number=last_segment.page_number))
        left.code(last_segment.source_text)
        right.caption(_t("saved_translation_caption"))
        right.write(last_translation.translated_text)

        rows = [
            {
                "#": index,
                _t("segment_col"): segment.segment_id,
                _t("page_col"): segment.page_number,
                _t("source_col"): _short_text(segment.source_text, 240),
                _t("translation_col"): _short_text(translation.translated_text, 520),
                _t("confidence_col"): translation.confidence,
            }
            for index, (segment, translation) in enumerate(translated_pairs, start=1)
        ]
        st.markdown(_t("all_saved_translations"))
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            height=min(520, 140 + 32 * min(len(rows), 12)),
        )

        full_text = _joined_translation_text(translated_pairs)
        st.text_area(
            _t("joined_live_text"),
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
    st.warning(_t("resume_warning", done=len(translations), total=len(segments)))
    if st.button(_t("resume_button"), type="primary", icon=":material/play_arrow:"):
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
        st.session_state["last_error"] = _t("empty_checkpoint")
        return

    source_path = Path(state.get("source_pdf_path", ""))
    if not source_path.exists():
        st.session_state["last_error"] = f"Nie mogę wznowić: brakuje pliku źródłowego `{source_path}`."
        return

    errors = _validate_provider_configuration(config.translator_provider, config.reviewer_provider)
    if errors:
        st.session_state["last_error"] = "\n".join(errors)
        return

    status = st.status(_t("resume_status"), expanded=True)
    progress_bar = status.progress(0, text=_t("resume_progress"))
    detail_slot = status.empty()
    history_slot = status.container(height=180, border=True)
    preview_slot = st.empty()
    _render_translation_preview(
        state,
        title=_t("checkpoint_before_resume"),
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
        _status_update(status, label=_t("resume_interrupted"), state="error")
        return

    st.session_state["translation_state"] = resumed_state
    st.session_state["loaded_checkpoint_job_id"] = None
    st.session_state["last_error"] = None
    if resumed_state.get("status") == "needs_human_review":
        _status_update(status, label=_t("needs_human_review"), state="complete")
    else:
        _status_update(status, label=_t("resume_done"), state="complete")


def _render_translation_preview(
    state: dict,
    *,
    title: str | None = None,
    max_rows: int | None = None,
    compact: bool = False,
) -> None:
    if title is None:
        title = _t("translation_preview")

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
        st.info(_t("empty_checkpoint"))
        return

    if not translated_pairs:
        st.info(_t("no_translation_yet", total=len(segments)))
        return

    st.caption(_t("saved_translation_count", done=len(translated_pairs), total=len(segments)))

    visible_pairs = translated_pairs if max_rows is None else translated_pairs[-max_rows:]
    rows = []
    for segment, translation in visible_pairs:
        rows.append(
            {
                _t("segment_col"): segment.segment_id,
                _t("page_col"): segment.page_number,
                _t("type_col"): segment.block_type,
                _t("source_col"): _short_text(segment.source_text, 220 if compact else 500),
                _t("translation_col"): _short_text(translation.translated_text, 320 if compact else 900),
                _t("confidence_col"): translation.confidence,
            }
        )

    st.dataframe(rows, hide_index=True, width="stretch")
    if not compact:
        full_text = _joined_translation_text(translated_pairs)
        st.text_area(
            _t("joined_saved_text"),
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
        parts.append(f"[{_t('page_col')} {page_number} | {segment_id}]\n{translated_text}")
    return "\n\n".join(parts)


def _text_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _render_human_review(state: dict) -> None:
    unresolved_segments = state.get("unresolved_segments", [])
    if not unresolved_segments:
        return

    segments = {segment.segment_id: segment for segment in state.get("segments", [])}
    translations = state.get("translations", {})

    st.subheader(_t("review_decisions"))
    with st.container(border=True):
        st.write(_t("bulk_review_note"))
        st.warning(_t("bulk_review_warning"))

        with st.container(horizontal=True):
            if st.button(
                _t("accept_all"),
                type="secondary",
                icon=":material/done_all:",
            ):
                decisions = {segment_id: {"action": "accept"} for segment_id in unresolved_segments}
                status = st.status(_t("rendering_after_accept"), expanded=True)
                progress_bar = status.progress(0, text=_t("rendering_start"))
                detail_slot = status.empty()
                history_slot = status.container(height=140, border=True)
                progress_callback = _make_progress_callback(status, progress_bar, detail_slot, history_slot)
                final_state = finalize_with_operator_decisions(state, decisions, progress_callback=progress_callback)
                _status_update(status, label=_t("pdf_after_accept"), state="complete")
                st.session_state["translation_state"] = final_state
                st.session_state["loaded_checkpoint_job_id"] = None
                st.rerun()

            if st.button(_t("hide_and_restart"), icon=":material/refresh:"):
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
        _t("issue_filter"),
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
        st.info(_t("no_segments_filter"))
        return

    max_to_show = st.slider(
        _t("segments_to_show"),
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
            left.markdown(_t("source"))
            left.code(segment.source_text)
            right.markdown(_t("translation"))
            edited = right.text_area(
                _t("approved_text"),
                value=translation.translated_text,
                key=f"edit_{segment_id}",
            )

            action = st.segmented_control(
                _t("decision"),
                ["edit", "accept", "keep_source"],
                default="edit",
                format_func={
                    "edit": _t("action_edit"),
                    "accept": _t("action_accept"),
                    "keep_source": _t("action_keep_source"),
                }.get,
                key=f"action_{segment_id}",
            )

            if issues:
                st.markdown(_t("issues"))
                for issue in issues:
                    st.warning(_issue_label(issue))

            decisions[segment_id] = {"action": action or "edit", "text": edited}

    with st.container(horizontal=True):
        if st.button(_t("save_decisions"), type="primary", icon=":material/save:"):
            status = st.status(_t("saving_decisions"), expanded=True)
            progress_bar = status.progress(0, text=_t("rendering_start"))
            detail_slot = status.empty()
            history_slot = status.container(height=140, border=True)
            progress_callback = _make_progress_callback(status, progress_bar, detail_slot, history_slot)
            final_state = finalize_with_operator_decisions(state, decisions, progress_callback=progress_callback)
            _status_update(status, label=_t("decisions_saved"), state="complete")
            st.session_state["translation_state"] = final_state
            st.session_state["loaded_checkpoint_job_id"] = None
            st.rerun()

        st.caption(_t("showing_segments", shown=min(max_to_show, len(filtered_segments)), total=len(filtered_segments)))


def _render_outputs(state: dict) -> None:
    if not state.get("output_pdf_path"):
        return

    output_path = Path(state["output_pdf_path"])
    report_path = Path(state["report_path"]) if state.get("report_path") else None
    st.success(_t("output_ready"))
    with st.container(horizontal=True):
        st.download_button(
            _t("download_pdf"),
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
        )
        if report_path and report_path.exists():
            st.download_button(
                _t("download_report"),
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="application/json",
                icon=":material/download:",
            )


st.set_page_config(page_title="Technical PDF Translator", layout="wide")
_init_session_state()

openai_ready = _has_env_value("OPENAI_API_KEY")
anthropic_ready = _has_env_value("ANTHROPIC_API_KEY")

with st.sidebar:
    st.header(_t("configuration"))
    st.segmented_control(
        _t("ui_language"),
        ["pl", "en"],
        default="pl",
        format_func={"pl": "PL", "en": "EN"}.get,
        key="ui_language",
        width="stretch",
        persist_state="session",
    )
    st.caption(
        _t(
            "openai_status",
            status=_t("ready") if openai_ready else _t("missing_key", key="OPENAI_API_KEY"),
        )
    )
    st.caption(
        _t(
            "anthropic_status",
            status=_t("ready") if anthropic_ready else _t("missing_key", key="ANTHROPIC_API_KEY"),
        )
    )

    st.subheader(_t("translation_languages"))
    source_language_options = _language_options_with_current(st.session_state.get("source_language"))
    target_language_options = _language_options_with_current(st.session_state.get("target_language"))
    source_language = st.selectbox(
        _t("source_language"),
        source_language_options,
        index=_language_option_index(source_language_options, "English"),
        key="source_language",
        accept_new_options=True,
        filter_mode="fuzzy",
        placeholder=_t("language_placeholder"),
        help=_t("language_help"),
        persist_state="session",
    )
    target_language = st.selectbox(
        _t("target_language"),
        target_language_options,
        index=_language_option_index(target_language_options, "Polish"),
        key="target_language",
        accept_new_options=True,
        filter_mode="fuzzy",
        placeholder=_t("language_placeholder"),
        help=_t("language_help"),
        persist_state="session",
    )
    if _normalized_language(source_language) == _normalized_language(target_language):
        st.warning(_t("same_language_warning"))

    mode = st.segmented_control(
        _t("mode"),
        ["standard", "high_assurance", "strict_regulatory"],
        default="standard",
        format_func=_mode_format,
        key="mode",
        width="stretch",
    )
    translator_provider = st.segmented_control(
        _t("translator"),
        ["openai", "mock"],
        default=_default_translator_provider(),
        format_func=_provider_format,
        key="translator_provider",
        width="stretch",
    )
    reviewer_provider = st.segmented_control(
        _t("reviewer"),
        ["anthropic", "openai", "mock"],
        default=_default_reviewer_provider(),
        format_func=_provider_format,
        key="reviewer_provider",
        width="stretch",
    )
    require_human_review = st.toggle(_t("require_human_review"), value=True, key="require_human_review")
    debug = st.toggle(
        _t("debug"),
        value=_default_debug_enabled(),
        key="debug",
        help=_t("debug_help"),
    )

st.title(_t("page_title"))
st.caption(_t("page_caption"))

if translator_provider == "mock":
    st.warning(_t("mock_warning"))

with st.form("translation_form"):
    uploaded = st.file_uploader(_t("upload_pdf"), type=["pdf"])
    submitted = st.form_submit_button(_t("start_translation"), type="primary", icon=":material/translate:")

if submitted:
    if uploaded is None:
        st.error(_t("select_pdf_first"))
    else:
        _run_translation(
            uploaded,
            source_language or "English",
            target_language or "Polish",
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
        st.info(_t("loaded_checkpoint", job_id=state.get("job_id"), status=state.get("status", "unknown")))
    _render_status(state)
    _render_checkpoint_resume(state)
    _render_translation_preview(state)
    _render_human_review(state)
    _render_outputs(state)
else:
    st.info(_t("empty_info"))
