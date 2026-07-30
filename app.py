from __future__ import annotations

import hashlib
import os
import re
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from itertools import count
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from translator.domain.glossary import load_glossary
from translator.domain.protected import validate_segment_invariants
from translator.languages import LANGUAGE_OPTIONS, language_index
from translator.llm.clients import build_reviewer, build_translator
from translator.schemas import JobConfig, ReviewFinding, TranslationResult
from translator.storage import JobStore
from translator.utils import new_job_id
from translator.workflow import finalize_with_operator_decisions, resume_mvp_pipeline, run_mvp_pipeline


load_dotenv(override=True)

STORAGE_INPUT = Path("storage/input")
STORAGE_OUTPUT = Path("storage/output")
PACKAGE_NAME = "tech-translator-agent"
WIDGET_KEY_COUNTER = count()

PROVIDER_LABELS = {
    "pl": {
        "openai": "OpenAI - realne tłumaczenie",
        "anthropic": "Anthropic - realna recenzja",
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
        "page_title": "TechnicAl. PDF Translator -- delta version",
        "page_caption": "PDF -> segmenty -> tłumaczenie -> walidacja -> recenzja -> decyzje operatora -> PDF.",
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
        "parallelism": "Równoległość",
        "translation_concurrency": "Tłumaczenie",
        "review_concurrency": "Recenzja",
        "concurrency_help": "Ile requestów LLM wysyłać naraz. Wyższa wartość zwykle przyspiesza długi PDF, ale może trafić w limity API.",
        "require_human_review": "Zatrzymaj przy istotnych zastrzeżeniach",
        "debug": "Tryb debug - logi w terminalu",
        "debug_help": "Wypisuje do konsoli każdy etap, segment, request LLM, czas trwania i błędy. Nie loguje kluczy API.",
        "mock_warning": "Wybrany jest mock. To sprawdza przepływ aplikacji, ale nie tłumaczy prawdziwego dokumentu.",
        "upload_pdf": "Wybierz cyfrowy PDF",
        "start_translation": "Rozpocznij tłumaczenie",
        "select_pdf_first": "Najpierw wybierz PDF.",
        "empty_info": "Wrzuć PDF i uruchom workflow. Jeśli masz klucz w `.env`, domyślnie użyję OpenAI zamiast mocka.",
        "processing": "Przetwarzam dokument...",
        "processing_detail": "Ekstrakcja PDF, segmentacja, tłumaczenie, walidacja i recenzja.",
        "debug_enabled": "Tryb debug jest włączony - szczegółowe logi lecą do terminala i `storage/logs/`.",
        "workflow_start": "Startuję workflow...",
        "interrupted": "Przetwarzanie przerwane",
        "needs_human_review": "Wymagana decyzja operatora",
        "processed": "Dokument przetworzony",
        "status": "Status",
        "translator_provider": "tłumacz",
        "review_provider": "recenzent",
        "parallelism_status": "równolegle",
        "source_to_target": "języki",
        "pipeline_progress_caption": "Postęp pipeline: tłumaczenie `{translations}/{segments}`, recenzja `{reviews}/{segments}`.",
        "translation_progress_metric": "Tłumaczenie",
        "progress_to_pdf": "Postęp do PDF: tłumaczenie `{translation_percent}%` | recenzja `{review_percent}%`.",
        "translation_progress_bar": "Tłumaczenie {percent}% ({done}/{total})",
        "review_progress_bar": "Recenzja {percent}% ({done}/{total})",
        "pdf_progress_delta": "T {translation_percent}% | R {review_percent}%",
        "pdf_status_waiting": "czeka",
        "pdf_status_progress": "w toku",
        "pdf_status_needs_decisions": "decyzje",
        "pdf_status_ready_to_generate": "generuj",
        "pdf_status_download": "pobierz",
        "pdf_status_missing": "brak pliku",
        "review_findings_caption": "Uwagi recenzenta: `{review_findings}`.",
        "review_findings_empty": "Brak zapisanych uwag recenzenta.",
        "review_finding_col_severity": "waga",
        "review_finding_col_category": "typ",
        "review_finding_col_source": "fragment źródła",
        "review_finding_col_translation": "fragment tłumaczenia",
        "review_finding_col_explanation": "uwaga",
        "review_finding_col_proposal": "propozycja poprawki",
        "review_finding_col_confidence": "pewność recenzji",
        "parallelism_change_pending": (
            "Suwaki UI ustawione na `{ui_translation}/{ui_review}`, ale aktualny checkpoint joba ma "
            "`{job_translation}/{job_review}`. Zmiana zadziała przy następnym wznowieniu albo nowym zadaniu; "
            "już uruchomionych requestów nie da się rozszerzyć w locie."
        ),
        "resume_parallelism_applied": "Wznowienie używa równoległości z UI: tłumaczenie `{translation}`, recenzja `{review}`.",
        "debug_log": "Debug log: `{path}`",
        "segments_metric": "Segmenty",
        "validation_metric": "Uwagi walidacji",
        "review_metric": "Recenzja",
        "decisions_metric": "Do decyzji",
        "pdf_metric": "PDF",
        "yes": "tak",
        "no": "nie",
        "cache_status": "Cache tłumaczeń: job `{job_hits}`, trwały `{persistent_hits}`, świeże requesty LLM `{llm_calls}`.",
        "token_usage_header": "Zużycie tokenów LLM",
        "llm_requests_metric": "Requesty LLM",
        "input_tokens_metric": "Input tokens",
        "output_tokens_metric": "Output tokens",
        "total_tokens_metric": "Total tokens",
        "no_token_usage_yet": "Tokeny pojawią się po zakończeniu pierwszego realnego requestu LLM. Cache i mock nie doliczają tokenów.",
        "llm_inflight": "Request LLM w toku: `{operation}` dla segmentu `{segment_id}` (`{provider}` `{model}`). Tokeny zaktualizują się po odpowiedzi providera.",
        "llm_inflight_many": "Requesty LLM w toku: `{active}` (`{operation}`, `{provider}` `{model}`), segmenty: {segments}. Tokeny zaktualizują się po odpowiedziach providera.",
        "llm_pipeline_inflight": "Pipeline LLM w toku: `{active}` requestów równolegle, segmenty: {segments}. Tokeny zaktualizują się po odpowiedziach providerów.",
        "llm_request_inflight_delta": "+1 w toku",
        "llm_requests_inflight_delta": "+{active} w toku",
        "estimated_input_delta": "+~{tokens} input w toku",
        "estimated_total_delta": "+~{tokens} szac. w toku",
        "estimated_token_note": "Wartość w toku jest lokalnym szacunkiem input promptu. Finalne input/output/total po odpowiedzi podaje provider.",
        "parallel_tasks_header": "Równoległe zadania LLM",
        "parallel_tasks_caption": "To są requesty aktualnie wysłane do providera. Lista odświeża się po checkpointach.",
        "parallel_task_status": "w toku",
        "parallel_task_col_slot": "#",
        "parallel_task_col_status": "status",
        "parallel_task_col_operation": "operacja",
        "parallel_task_col_segment": "segment",
        "parallel_task_col_page": "strona",
        "parallel_task_col_provider": "provider",
        "parallel_task_col_model": "model",
        "parallel_task_col_estimated_input": "~input tokens",
        "parallel_task_col_source": "fragment źródła",
        "progress_stage": "Etap",
        "progress_segment": "segment",
        "progress_stage_progress": "postęp etapu",
        "pipeline_split_progress": "tłumaczenie {translations_done}/{translations_total} | recenzja {reviews_done}/{reviews_total}",
        "progress_working": "Pracuję...",
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
        "resume_warning": "Mam zapisany checkpoint tego joba. Jeśli TechnicAl się przeładował albo przerwał pracę, możesz kontynuować bez tracenia zapisanych segmentów ({done}/{total}).",
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
        "bulk_review_note": "Nie pokazuję wszystkich uwag naraz, bo przy dużym PDF-ie UI staje się nieużywalny. Możesz przejrzeć próbkę albo świadomie zaakceptować aktualne tłumaczenia zbiorczo.",
        "bulk_review_warning": "Akcja zbiorcza ma sens tylko po realnym tłumaczeniu. Po trybie mock wygeneruje PDF z wieloma fragmentami po angielsku.",
        "accept_all": "Zaakceptuj wszystkie aktualne i wygeneruj PDF",
        "rendering_after_accept": "Generuję PDF po zbiorczej akceptacji...",
        "rendering_start": "Startuję renderowanie...",
        "pdf_after_accept": "PDF wygenerowany po zbiorczej akceptacji",
        "pdf_ready_to_generate": "Tłumaczenie i recenzja są gotowe. Możesz teraz wygenerować PDF.",
        "generate_pdf": "Generuj PDF",
        "generating_pdf": "Generuję PDF...",
        "pdf_generated": "PDF wygenerowany",
        "hide_and_restart": "Ukryj wynik w tej sesji i zacznij od nowa",
        "issue_filter": "Filtr uwag",
        "no_segments_filter": "Brak segmentów dla wybranego filtra.",
        "segments_to_show": "Ile segmentów pokazać",
        "source": "Źródło",
        "translation": "Tłumaczenie",
        "current_translation": "Aktualne tłumaczenie",
        "approved_text": "Zatwierdzony tekst",
        "operator_refinement_title": "Poprawka operatora z przerobieniem zdania",
        "operator_refinement_caption": "Wpisz preferowaną frazę/termin. TechnicAl przerobi cały segment tak, żeby fraza pasowała składniowo i semantycznie do zdania.",
        "operator_replace_from": "Fragment do zastąpienia",
        "operator_replace_placeholder": "np. wysuszonej warstwie farby",
        "operator_preferred_phrase": "Lepsza fraza / termin bazowy",
        "operator_preferred_placeholder": "np. wyschnięta powłoka farby",
        "operator_rewrite_button": "Przerób segment z tą frazą",
        "operator_rewrite_missing_phrase": "Wpisz lepszą frazę lub termin bazowy.",
        "operator_rewrite_spinner": "Przerabiam segment i sprawdzam wynik...",
        "operator_rewrite_queued": "Poprawka została wysłana do tła. Możesz kontynuować review innych segmentów.",
        "operator_rewrite_background_running": "Poprawka działa w tle dla frazy `{phrase}`. Możesz kontynuować review.",
        "operator_rewrite_refresh": "Odśwież wynik poprawki",
        "operator_rewrite_failed": "Poprawka w tle zakończyła się błędem: {error}",
        "operator_rewrite_result": "Wynik poprawki operatora",
        "operator_rewrite_applied": "Wynik został podstawiony do pola „Zatwierdzony tekst”.",
        "operator_rewrite_not_auto_applied": "Nie podstawiono automatycznie, bo pole „Zatwierdzony tekst” zmieniło się podczas pracy w tle.",
        "operator_rewrite_apply_result": "Wstaw wynik do zatwierdzonego tekstu",
        "operator_rewrite_source": "Tryb poprawki: `{mode}`.",
        "operator_rewrite_mode_llm": "LLM + review",
        "operator_rewrite_mode_local": "lokalna podmiana",
        "operator_rewrite_local_warning": "To była tylko lokalna podmiana tekstu. Dla składniowej odmiany frazy wybierz realnego tłumacza OpenAI.",
        "operator_rewrite_check_ok": "Kontrola po poprawce: brak krytycznych problemów walidacji; recenzent nie zgłasza krytycznych ani ważnych zastrzeżeń.",
        "operator_rewrite_check_warning": "Kontrola po poprawce nadal zgłasza zastrzeżenia — obejrzyj je przed akceptacją.",
        "operator_rewrite_review_unavailable": "Nie uruchomiłem realnego review tej poprawki. Sprawdzona jest tylko walidacja wartości chronionych.",
        "operator_rewrite_review_verdict": "Werdykt recenzenta dla poprawki: `{verdict}`.",
        "operator_rewrite_validation_issues": "Uwagi walidacji po poprawce",
        "operator_rewrite_review_findings": "Uwagi recenzenta po poprawce",
        "operator_refinement_note": "Fragment poprawiony przez użytkownika: preferowana fraza `{phrase}`.",
        "decision": "Decyzja",
        "action_edit": "Zapisz edycję",
        "action_accept": "Akceptuj",
        "action_keep_source": "Zostaw źródło",
        "issues": "Uwagi i zastrzeżenia",
        "save_decisions": "Zapisz decyzje dla pokazanych segmentów",
        "saving_decisions": "Zapisuję decyzje i generuję wynik...",
        "decisions_saved": "Decyzje zapisane i wynik wygenerowany",
        "showing_segments": "Pokazuję {shown} z {total} segmentów po filtrze.",
        "output_ready": "PDF wynikowy jest gotowy.",
        "output_pdf_missing": "Checkpoint wskazuje PDF, ale plik nie istnieje na dysku: `{path}`.",
        "download_pdf": "Pobierz PDF",
        "download_report": "Pobierz raport JSON",
        "loaded_checkpoint": "Wczytałem ostatni zapisany checkpoint: `{job_id}` ze statusem `{status}`.",
        "checkpoint_load_error": "Nie udało się wczytać checkpointu: {error}",
        "resume_missing_source": "Nie mogę wznowić: brakuje pliku źródłowego `{path}`.",
        "review_expander_label": "{segment_id} - strona {page_number}",
        "segment_col": "segment",
        "page_col": "strona",
        "type_col": "typ",
        "source_col": "źródło",
        "translation_col": "tłumaczenie",
        "confidence_col": "pewność",
        "missing_openai_translator": "Brakuje `OPENAI_API_KEY` w `.env`, a wybrano tłumacza OpenAI.",
        "missing_openai_reviewer": "Brakuje `OPENAI_API_KEY` w `.env`, a wybrano recenzenta OpenAI.",
        "missing_anthropic_reviewer": "Brakuje `ANTHROPIC_API_KEY` w `.env`, a wybrano recenzenta Anthropic.",
    },
    "en": {
        "page_title": "TechnicAl. PDF Translator -- delta version",
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
        "parallelism": "Parallelism",
        "translation_concurrency": "Translation",
        "review_concurrency": "Review",
        "concurrency_help": "How many LLM requests to send at the same time. Higher values usually speed up long PDFs, but may hit API limits.",
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
        "parallelism_status": "parallel",
        "source_to_target": "languages",
        "pipeline_progress_caption": "Pipeline progress: translations `{translations}/{segments}`, review `{reviews}/{segments}`.",
        "translation_progress_metric": "Translation",
        "progress_to_pdf": "PDF progress: translation `{translation_percent}%` | review `{review_percent}%`.",
        "translation_progress_bar": "Translation {percent}% ({done}/{total})",
        "review_progress_bar": "Review {percent}% ({done}/{total})",
        "pdf_progress_delta": "T {translation_percent}% | R {review_percent}%",
        "pdf_status_waiting": "waiting",
        "pdf_status_progress": "progress",
        "pdf_status_needs_decisions": "decisions",
        "pdf_status_ready_to_generate": "generate",
        "pdf_status_download": "download",
        "pdf_status_missing": "missing file",
        "review_findings_caption": "Review findings: `{review_findings}`.",
        "review_findings_empty": "No reviewer findings saved yet.",
        "review_finding_col_severity": "severity",
        "review_finding_col_category": "type",
        "review_finding_col_source": "source evidence",
        "review_finding_col_translation": "translation evidence",
        "review_finding_col_explanation": "review note",
        "review_finding_col_proposal": "proposed fix",
        "review_finding_col_confidence": "review confidence",
        "parallelism_change_pending": (
            "The UI sliders are set to `{ui_translation}/{ui_review}`, but this job checkpoint has "
            "`{job_translation}/{job_review}`. The change applies on the next resume or new job; "
            "already running requests cannot be expanded in flight."
        ),
        "resume_parallelism_applied": "Resume uses UI parallelism: translation `{translation}`, review `{review}`.",
        "debug_log": "Debug log: `{path}`",
        "segments_metric": "Segments",
        "validation_metric": "Validation",
        "review_metric": "Review",
        "decisions_metric": "To decide",
        "pdf_metric": "PDF",
        "yes": "yes",
        "no": "no",
        "cache_status": "Translation cache: job `{job_hits}`, persistent `{persistent_hits}`, fresh LLM requests `{llm_calls}`.",
        "token_usage_header": "LLM token usage",
        "llm_requests_metric": "LLM requests",
        "input_tokens_metric": "Input tokens",
        "output_tokens_metric": "Output tokens",
        "total_tokens_metric": "Total tokens",
        "no_token_usage_yet": "Tokens will appear after the first real LLM request finishes. Cache and mock do not add tokens.",
        "llm_inflight": "LLM request in progress: `{operation}` for segment `{segment_id}` (`{provider}` `{model}`). Token usage will update after the provider responds.",
        "llm_inflight_many": "LLM requests in progress: `{active}` (`{operation}`, `{provider}` `{model}`), segments: {segments}. Token usage will update after provider responses.",
        "llm_pipeline_inflight": "LLM pipeline in progress: `{active}` parallel requests, segments: {segments}. Token usage will update after provider responses.",
        "llm_request_inflight_delta": "+1 in flight",
        "llm_requests_inflight_delta": "+{active} in flight",
        "estimated_input_delta": "+~{tokens} input in flight",
        "estimated_total_delta": "+~{tokens} est. in flight",
        "estimated_token_note": "The in-flight value is a local input-prompt estimate. Final input/output/total comes from the provider response.",
        "parallel_tasks_header": "Parallel LLM tasks",
        "parallel_tasks_caption": "These are requests currently sent to the provider. The list refreshes on checkpoints.",
        "parallel_task_status": "in flight",
        "parallel_task_col_slot": "#",
        "parallel_task_col_status": "status",
        "parallel_task_col_operation": "operation",
        "parallel_task_col_segment": "segment",
        "parallel_task_col_page": "page",
        "parallel_task_col_provider": "provider",
        "parallel_task_col_model": "model",
        "parallel_task_col_estimated_input": "~input tokens",
        "parallel_task_col_source": "source preview",
        "progress_stage": "Stage",
        "progress_segment": "segment",
        "progress_stage_progress": "stage progress",
        "pipeline_split_progress": "translations {translations_done}/{translations_total} | review {reviews_done}/{reviews_total}",
        "progress_working": "Working...",
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
        "resume_warning": "This job has a saved checkpoint. If TechnicAl reloaded or work was interrupted, you can continue without losing saved segments ({done}/{total}).",
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
        "pdf_ready_to_generate": "Translation and review are complete. You can generate the PDF now.",
        "generate_pdf": "Generate PDF",
        "generating_pdf": "Generating PDF...",
        "pdf_generated": "PDF generated",
        "hide_and_restart": "Hide this result in this session and start again",
        "issue_filter": "Issue filter",
        "no_segments_filter": "No segments for the selected filter.",
        "segments_to_show": "How many segments to show",
        "source": "Source",
        "translation": "Translation",
        "current_translation": "Current translation",
        "approved_text": "Approved text",
        "operator_refinement_title": "Operator phrase refinement",
        "operator_refinement_caption": "Enter the preferred phrase/term. TechnicAl rewrites the whole segment so the phrase fits the sentence syntactically and semantically.",
        "operator_replace_from": "Text to replace",
        "operator_replace_placeholder": "e.g. dried ink layer",
        "operator_preferred_phrase": "Better phrase / base term",
        "operator_preferred_placeholder": "e.g. dry ink film",
        "operator_rewrite_button": "Rewrite segment with this phrase",
        "operator_rewrite_missing_phrase": "Enter a better phrase or base term.",
        "operator_rewrite_spinner": "Rewriting the segment and checking the result...",
        "operator_rewrite_queued": "The refinement was sent to the background. You can continue reviewing other segments.",
        "operator_rewrite_background_running": "Refinement is running in the background for phrase `{phrase}`. You can continue reviewing.",
        "operator_rewrite_refresh": "Refresh refinement result",
        "operator_rewrite_failed": "Background refinement failed: {error}",
        "operator_rewrite_result": "Operator refinement result",
        "operator_rewrite_applied": "The result was copied into the Approved text field.",
        "operator_rewrite_not_auto_applied": "I did not auto-apply it because the Approved text field changed while the background task was running.",
        "operator_rewrite_apply_result": "Insert result into approved text",
        "operator_rewrite_source": "Refinement mode: `{mode}`.",
        "operator_rewrite_mode_llm": "LLM + review",
        "operator_rewrite_mode_local": "local replacement",
        "operator_rewrite_local_warning": "This was only a local text replacement. Choose the real OpenAI translator for grammatical inflection of the phrase.",
        "operator_rewrite_check_ok": "Post-refinement check: no critical validation issues; the reviewer reports no critical or major findings.",
        "operator_rewrite_check_warning": "The post-refinement check still reports issues — review them before accepting.",
        "operator_rewrite_review_unavailable": "I did not run a real review for this refinement. Only protected-value validation was checked.",
        "operator_rewrite_review_verdict": "Reviewer verdict for this refinement: `{verdict}`.",
        "operator_rewrite_validation_issues": "Validation issues after refinement",
        "operator_rewrite_review_findings": "Reviewer findings after refinement",
        "operator_refinement_note": "User-refined fragment: preferred phrase `{phrase}`.",
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
        "output_pdf_missing": "The checkpoint points to a PDF, but the file is missing on disk: `{path}`.",
        "download_pdf": "Download PDF",
        "download_report": "Download JSON report",
        "loaded_checkpoint": "Loaded the latest checkpoint: `{job_id}` with status `{status}`.",
        "checkpoint_load_error": "Could not load checkpoint: {error}",
        "resume_missing_source": "Cannot resume: source file is missing: `{path}`.",
        "review_expander_label": "{segment_id} - page {page_number}",
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

OPERATION_LABELS_PL = {
    "translate": "tłumaczenie",
    "review": "recenzja",
    "revise": "poprawka",
    "pipeline": "pipeline",
    "llm": "LLM",
}

ISSUE_TYPE_LABELS_PL = {
    "missing_number": "brak liczby",
    "changed_number": "zmieniona liczba",
    "missing_unit": "brak jednostki",
    "changed_unit": "zmieniona jednostka",
    "missing_reference": "brak odwołania",
    "changed_comparator": "zmieniony operator",
    "missing_negation": "brak negacji",
    "table_alignment": "układ tabeli",
    "untranslated_fragment": "nieprzetłumaczony fragment",
    "unexpected_addition": "nieoczekiwany dodatek",
    "forbidden_term": "termin zabroniony",
    "unchanged_source": "pozostawione źródło",
    "terminology": "terminologia",
    "chemical_terminology": "terminologia chemiczna",
    "regulatory_terminology": "terminologia regulacyjna",
    "omission": "pominięcie",
    "addition": "dodana informacja",
    "meaning": "znaczenie",
    "number": "liczba",
    "unit": "jednostka",
    "negation": "negacja",
    "table_relationship": "relacja w tabeli",
    "inconsistency": "niespójność",
    "manual_review": "do ręcznej oceny",
    "issue": "uwaga",
}

SEVERITY_LABELS_PL = {
    "critical": "KRYTYCZNE",
    "major": "WAŻNE",
    "minor": "DROBNE",
    "warning": "OSTRZEŻENIE",
    "style": "STYL",
    "info": "INFO",
    "manual_review": "RĘCZNA OCENA",
}


def _init_session_state() -> None:
    st.session_state.setdefault("last_error", None)
    if _is_stale_preview_key_error(st.session_state.get("last_error")):
        st.session_state["last_error"] = None

    if "translation_state" not in st.session_state:
        latest_state = _load_latest_checkpoint(silent=True)
        st.session_state["translation_state"] = latest_state
        st.session_state["loaded_checkpoint_job_id"] = latest_state.get("job_id") if latest_state else None
    else:
        st.session_state.setdefault("loaded_checkpoint_job_id", None)


def _ui_language() -> str:
    value = str(st.session_state.get("ui_language") or "pl").lower()
    return value if value in UI_TEXT else "pl"


def _is_stale_preview_key_error(error: object) -> bool:
    text = str(error or "")
    return (
        "multiple elements with the same key" in text
        and (
            "live_joined_translation" in text
            or "checkpoint_joined_translation" in text
        )
    )


def _t(text_key: str, **kwargs: object) -> str:
    template = UI_TEXT.get(_ui_language(), UI_TEXT["pl"]).get(text_key, UI_TEXT["pl"].get(text_key, text_key))
    return template.format(**kwargs) if kwargs else template


def _app_version() -> str:
    env_version = os.getenv("TECHNICAL_APP_VERSION", "").strip()
    if env_version:
        return env_version

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        pass

    try:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        return str(pyproject.get("project", {}).get("version") or "dev")
    except Exception:
        return "dev"


def _progress_message(raw_message: str) -> str:
    if _ui_language() != "en":
        return raw_message

    exact = {
        "Pracuję...": "Working...",
        "Ekstrahuję tekst i tabele z PDF-a": "Extracting text and tables from the PDF",
        "Przygotowuję segmenty i chronione wartości": "Preparing segments and protected values",
        "Pomijam już przetłumaczony segment": "Skipping already translated segment",
        "Używam zapisanego tłumaczenia identycznego segmentu": "Reusing an identical segment translation",
        "Używam trwałego cache tłumaczenia": "Using persistent translation cache",
        "Pipeline: pomijam już przetłumaczony segment": "Pipeline: skipping already translated segment",
        "Pipeline: używam zapisanego tłumaczenia identycznego segmentu": "Pipeline: reusing identical segment translation",
        "Pipeline: używam trwałego cache tłumaczenia": "Pipeline: using persistent translation cache",
        "Pipeline: tłumaczenie i review równolegle": "Pipeline: translating and reviewing in parallel",
        "Pipeline: tłumaczenie i recenzja równolegle": "Pipeline: translating and reviewing in parallel",
        "Pipeline: segment przetłumaczony": "Pipeline: segment translated",
        "Pipeline: segment po review": "Pipeline: segment reviewed",
        "Pipeline: segment po recenzji": "Pipeline: segment reviewed",
        "Tłumaczę segment": "Translating segment",
        "Tłumaczę segmenty równolegle": "Translating segments in parallel",
        "Segment przetłumaczony": "Segment translated",
        "Sprawdzam liczby, jednostki i odnośniki": "Checking numbers, units and references",
        "Pomijam segment już zrecenzowany w checkpoincie": "Skipping segment already reviewed in checkpoint",
        "Recenzuję segment": "Reviewing segment",
        "Recenzuję segmenty równolegle": "Reviewing segments in parallel",
        "Segment po review": "Segment reviewed",
        "Segment po recenzji": "Segment reviewed",
        "Rozstrzygam problemy i routing": "Resolving issues and routing",
        "Poprawiam zakwestionowany segment": "Revising flagged segment",
        "Generuję PDF wynikowy": "Generating output PDF",
        "Weryfikuję PDF wynikowy": "Verifying output PDF",
        "Zapisuję raport audytowy": "Writing audit report",
        "Workflow zakończony": "Workflow complete",
        "Wznawiam: ekstrahuję PDF": "Resuming: extracting PDF",
        "Wznawiam: przygotowuję segmenty": "Resuming: preparing segments",
    }
    if raw_message in exact:
        return exact[raw_message]

    patterns = [
        (r"^Ekstrakcja zakończona: (\d+) segmentów$", r"Extraction complete: \1 segments"),
        (r"^Wymagana decyzja operatora: (\d+) segmentów$", r"Operator decision required: \1 segments"),
        (r"^Waliduję tłumaczenia - cykl (\d+)/(\d+)$", r"Validating translations - cycle \1/\2"),
        (r"^Uruchamiam review - cykl (\d+)/(\d+)$", r"Running review - cycle \1/\2"),
        (r"^Uruchamiam recenzję - cykl (\d+)/(\d+)$", r"Running review - cycle \1/\2"),
        (r"^Review już gotowy z pipeline - cykl (\d+)/(\d+)$", r"Review already completed by pipeline - cycle \1/\2"),
        (r"^Recenzja już gotowa z pipeline - cykl (\d+)/(\d+)$", r"Review already completed by pipeline - cycle \1/\2"),
        (r"^Wznawiam tłumaczenie od checkpointu \((\d+)/(\d+)\)$", r"Resuming translation from checkpoint (\1/\2)"),
        (
            r"^Wznawiam pipeline tłumaczenie→review \((\d+)/(\d+) tłumaczeń, (\d+)/(\d+) review\)$",
            r"Resuming translation→review pipeline (\1/\2 translations, \3/\4 reviews)",
        ),
        (
            r"^Wznawiam pipeline tłumaczenie→recenzja \((\d+)/(\d+) tłumaczeń, (\d+)/(\d+) recenzji\)$",
            r"Resuming translation→review pipeline (\1/\2 translations, \3/\4 reviews)",
        ),
    ]
    for pattern, replacement in patterns:
        translated = re.sub(pattern, replacement, raw_message)
        if translated != raw_message:
            return translated

    return raw_message


def _pipeline_progress_label(event: dict) -> str | None:
    translations_total = _safe_int(event.get("translations_total"))
    reviews_total = _safe_int(event.get("reviews_total"))
    if translations_total <= 0 and reviews_total <= 0:
        return None

    return _t(
        "pipeline_split_progress",
        translations_done=_safe_int(event.get("translations_done")),
        translations_total=translations_total,
        reviews_done=_safe_int(event.get("reviews_done")),
        reviews_total=reviews_total,
    )


def _localized_issue_message(raw_message: str) -> str:
    if _ui_language() != "en":
        return raw_message

    exact = {
        "Tłumaczenie wygląda jak nieprzetłumaczony fragment źródłowy.": (
            "The translation appears to be an untranslated source fragment."
        ),
        "Fragment angielski został pozostawiony bez tłumaczenia.": (
            "The English fragment was left untranslated."
        ),
        "Źródło zawiera negację, której nie widać w tłumaczeniu.": (
            "The source contains a negation that is not visible in the translation."
        ),
    }
    if raw_message in exact:
        return exact[raw_message]

    patterns = [
        (
            r"^Chroniony element '(.*?)' \((.*?)\) nie występuje w tłumaczeniu\.$",
            r"Protected element '\1' (\2) is missing from the translation.",
        ),
        (
            r"^Tłumaczenie dodało chroniony element '(.*?)' \((.*?)\), którego nie było w źródle\.$",
            r"The translation added protected element '\1' (\2), which was not present in the source.",
        ),
        (
            r"^Termin '(.*?)' nie używa zatwierdzonego odpowiednika '(.*?)'\.$",
            r"Term '\1' does not use the approved equivalent '\2'.",
        ),
        (
            r"^Termin '(.*?)' jest zabroniony dla '(.*?)'\.$",
            r"Term '\1' is forbidden for '\2'.",
        ),
    ]
    for pattern, replacement in patterns:
        translated = re.sub(pattern, replacement, raw_message)
        if translated != raw_message:
            return translated

    return raw_message


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
            st.session_state["last_error"] = _t("checkpoint_load_error", error=exc)
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
    severity = _localized_severity(str(getattr(issue, "severity", "info")))
    issue_type = _localized_issue_type(str(getattr(issue, "issue_type", getattr(issue, "category", "issue"))))
    message = _localized_issue_message(str(getattr(issue, "message", getattr(issue, "explanation", ""))))
    return f"{severity} - {issue_type}: {message}"


def _review_findings_count(state: dict) -> int:
    return sum(len(getattr(result, "findings", []) or []) for result in state.get("review_results", {}).values())


def _review_findings_rows(state: dict) -> list[dict[str, object]]:
    segments_by_id = {segment.segment_id: segment for segment in state.get("segments", [])}
    translations = state.get("translations", {})
    rows = []

    for result in state.get("review_results", {}).values():
        result_segment_id = str(getattr(result, "segment_id", "") or "")
        for finding in getattr(result, "findings", []) or []:
            segment_id = str(getattr(finding, "segment_id", result_segment_id) or result_segment_id)
            segment = segments_by_id.get(segment_id)
            translation = translations.get(segment_id)
            source_evidence = str(getattr(finding, "source_evidence", "") or getattr(segment, "source_text", ""))
            translation_evidence = str(
                getattr(finding, "translation_evidence", "") or getattr(translation, "translated_text", "")
            )

            rows.append(
                {
                    "#": len(rows) + 1,
                    _t("segment_col"): segment_id or "-",
                    _t("page_col"): getattr(segment, "page_number", "-"),
                    _t("review_finding_col_severity"): _localized_severity(str(getattr(finding, "severity", "info"))),
                    _t("review_finding_col_category"): _localized_issue_type(
                        str(getattr(finding, "category", "issue"))
                    ),
                    _t("review_finding_col_source"): _short_text(source_evidence, 220),
                    _t("review_finding_col_translation"): _short_text(translation_evidence, 260),
                    _t("review_finding_col_explanation"): _short_text(
                        _localized_issue_message(str(getattr(finding, "explanation", ""))),
                        320,
                    ),
                    _t("review_finding_col_proposal"): _short_text(
                        str(getattr(finding, "proposed_translation", "") or ""),
                        260,
                    ),
                    _t("review_finding_col_confidence"): str(getattr(finding, "confidence", "-") or "-"),
                }
            )

    return list(reversed(rows))


def _render_review_findings_table(state: dict) -> None:
    rows = _review_findings_rows(state)
    if not rows:
        st.caption(_t("review_findings_empty"))
        return

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=min(520, 140 + 32 * min(len(rows), 12)),
    )


def _operator_refinement_result_key(segment_id: str) -> str:
    return f"operator_refinement_result_{segment_id}"


def _operator_refinement_task_key(segment_id: str) -> str:
    return f"operator_refinement_task_{segment_id}"


def _operator_refinement_error_key(segment_id: str) -> str:
    return f"operator_refinement_error_{segment_id}"


def _operator_refinement_worker_count() -> int:
    try:
        requested = int(os.getenv("OPERATOR_REFINEMENT_WORKERS", "3"))
    except ValueError:
        requested = 3
    return min(max(requested, 1), 8)


@st.cache_resource
def _operator_refinement_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_operator_refinement_worker_count(),
        thread_name_prefix="operator-refine",
    )


def _suggest_replacement_text(issues: list[object]) -> str:
    for issue in issues:
        evidence = str(getattr(issue, "translation_evidence", "") or getattr(issue, "translated_value", "") or "")
        evidence = " ".join(evidence.split())
        if evidence and len(evidence) <= 120:
            return evidence
    return ""


def _poll_operator_refinement_task(segment_id: str, *, edit_key: str, fallback_text: str) -> None:
    task_key = _operator_refinement_task_key(segment_id)
    task = st.session_state.get(task_key)
    if not isinstance(task, dict):
        return

    future = task.get("future")
    if not isinstance(future, Future) or not future.done():
        return

    st.session_state.pop(task_key, None)
    error_key = _operator_refinement_error_key(segment_id)
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001 - show background failure in the review UI
        st.session_state[error_key] = str(exc)
        return

    base_edit_value = str(task.get("edit_value_at_submit") or "")
    current_edit_value = str(st.session_state.get(edit_key, fallback_text) or "")
    auto_applied = current_edit_value == base_edit_value
    result["auto_applied_to_editor"] = auto_applied
    if auto_applied:
        st.session_state[edit_key] = str(result.get("proposed_text") or "")
    st.session_state[_operator_refinement_result_key(segment_id)] = result
    st.session_state.pop(error_key, None)


def _operator_refinement_running(segment_id: str) -> dict | None:
    task = st.session_state.get(_operator_refinement_task_key(segment_id))
    if not isinstance(task, dict):
        return None
    future = task.get("future")
    if isinstance(future, Future) and not future.done():
        return task
    return None


def _render_operator_phrase_refinement(
    segment: object,
    translation: TranslationResult,
    config: JobConfig,
    *,
    edit_key: str,
    issues: list[object],
) -> None:
    segment_id = str(getattr(segment, "segment_id", "segment"))
    replace_key = f"operator_replace_{segment_id}"
    phrase_key = f"operator_phrase_{segment_id}"
    result_key = _operator_refinement_result_key(segment_id)
    error_key = _operator_refinement_error_key(segment_id)

    st.session_state.setdefault(replace_key, _suggest_replacement_text(issues))
    st.session_state.setdefault(phrase_key, "")
    _poll_operator_refinement_task(segment_id, edit_key=edit_key, fallback_text=translation.translated_text)

    with st.container(border=True):
        st.markdown(f"##### :material/edit_note: {_t('operator_refinement_title')}")
        st.caption(_t("operator_refinement_caption"))

        replace_col, phrase_col = st.columns(2)
        replace_col.text_input(
            _t("operator_replace_from"),
            key=replace_key,
            placeholder=_t("operator_replace_placeholder"),
        )
        phrase_col.text_input(
            _t("operator_preferred_phrase"),
            key=phrase_key,
            placeholder=_t("operator_preferred_placeholder"),
        )

        running_task = _operator_refinement_running(segment_id)
        if st.button(
            _t("operator_rewrite_button"),
            key=f"operator_rewrite_{segment_id}",
            icon=":material/auto_fix_high:",
            disabled=running_task is not None,
        ):
            preferred_phrase = str(st.session_state.get(phrase_key, "") or "").strip()
            replace_phrase = str(st.session_state.get(replace_key, "") or "").strip()
            if not preferred_phrase:
                st.warning(_t("operator_rewrite_missing_phrase"))
            else:
                future = _operator_refinement_executor().submit(
                    _rewrite_translation_with_operator_phrase,
                    segment,
                    translation,
                    config,
                    preferred_phrase=preferred_phrase,
                    replace_phrase=replace_phrase,
                    issues=issues,
                )
                st.session_state[_operator_refinement_task_key(segment_id)] = {
                    "future": future,
                    "preferred_phrase": preferred_phrase,
                    "replace_phrase": replace_phrase,
                    "edit_value_at_submit": str(st.session_state.get(edit_key, translation.translated_text) or ""),
                }
                st.session_state.pop(result_key, None)
                st.session_state.pop(error_key, None)
                st.success(_t("operator_rewrite_queued"))
                _poll_operator_refinement_task(segment_id, edit_key=edit_key, fallback_text=translation.translated_text)
                running_task = _operator_refinement_running(segment_id)

        error = st.session_state.get(error_key)
        if error:
            st.error(_t("operator_rewrite_failed", error=error))

        if running_task:
            st.info(
                _t(
                    "operator_rewrite_background_running",
                    phrase=running_task.get("preferred_phrase", "-"),
                )
            )
            st.button(
                _t("operator_rewrite_refresh"),
                key=f"operator_rewrite_refresh_{segment_id}",
                icon=":material/refresh:",
            )

        result = st.session_state.get(result_key)
        if result:
            _render_operator_refinement_result(result, edit_key=edit_key)


def _rewrite_translation_with_operator_phrase(
    segment: object,
    translation: TranslationResult,
    config: JobConfig,
    *,
    preferred_phrase: str,
    replace_phrase: str,
    issues: list[object],
) -> dict[str, object]:
    glossary = load_glossary(config.glossary_path)
    mode = "local"

    if config.translator_provider == "openai" and _has_env_value("OPENAI_API_KEY"):
        mode = "llm"
        explanation_parts = [
            "The human operator requires a terminology refinement.",
            f"Preferred target-language base phrase: {preferred_phrase!r}.",
            "Rewrite the full target-language segment, not only the phrase.",
            "Use the preferred phrase exactly when grammatical; otherwise inflect it so the full sentence is syntactically natural.",
            "Preserve the source meaning and all protected values, units, identifiers and references.",
            "Add no explanations to the translated text.",
        ]
        if replace_phrase:
            explanation_parts.insert(2, f"Current target-language wording to replace: {replace_phrase!r}.")
        if issues:
            explanation_parts.append(
                "Existing review/validation context: "
                + "; ".join(_raw_issue_summary(issue) for issue in issues[:5])
            )
        operator_finding = ReviewFinding(
            segment_id=str(getattr(segment, "segment_id", translation.segment_id)),
            severity="major",
            category="operator_phrase_refinement",
            source_evidence=str(getattr(segment, "source_text", "")),
            translation_evidence=translation.translated_text,
            explanation=" ".join(explanation_parts),
            proposed_translation=None,
            confidence="high",
        )
        revised = build_translator(config).revise(
            segment,  # type: ignore[arg-type]
            translation,
            [operator_finding],
            glossary,
            config,
        )
    else:
        revised_text = _local_operator_phrase_rewrite(
            translation.translated_text,
            replace_phrase=replace_phrase,
            preferred_phrase=preferred_phrase,
        )
        revised = translation.model_copy(
            update={
                "translated_text": revised_text,
                "translator_notes": [
                    *translation.translator_notes,
                    f"Operator local phrase refinement: {preferred_phrase}",
                ],
                "confidence": "medium",
            }
        )

    validation_issues = validate_segment_invariants(segment, revised)  # type: ignore[arg-type]
    validation_issues.extend(
        glossary.validate_translation(
            str(getattr(segment, "segment_id", translation.segment_id)),
            str(getattr(segment, "source_text", "")),
            revised.translated_text,
            config.target_language,
        )
    )

    review_result = None
    if _reviewer_provider_ready(config):
        review_result = build_reviewer(config).review(segment, revised, glossary, config)  # type: ignore[arg-type]

    return {
        "mode": mode,
        "preferred_phrase": preferred_phrase,
        "replace_phrase": replace_phrase,
        "proposed_text": revised.translated_text,
        "validation_issues": validation_issues,
        "review_result": review_result,
        "translation_token_usage": revised.token_usage,
        "review_token_usage": getattr(review_result, "token_usage", None),
    }


def _local_operator_phrase_rewrite(text: str, *, replace_phrase: str, preferred_phrase: str) -> str:
    if replace_phrase:
        pattern = re.compile(re.escape(replace_phrase), flags=re.IGNORECASE)
        rewritten, count_replacements = pattern.subn(preferred_phrase, text, count=1)
        if count_replacements:
            return rewritten
    if preferred_phrase.lower() in text.lower():
        return text
    return text


def _reviewer_provider_ready(config: JobConfig) -> bool:
    if config.reviewer_provider == "openai":
        return _has_env_value("OPENAI_API_KEY")
    if config.reviewer_provider == "anthropic":
        return _has_env_value("ANTHROPIC_API_KEY")
    return False


def _raw_issue_summary(issue: object) -> str:
    severity = str(getattr(issue, "severity", "info"))
    issue_type = str(getattr(issue, "issue_type", getattr(issue, "category", "issue")))
    message = str(getattr(issue, "message", getattr(issue, "explanation", "")))
    return f"{severity} - {issue_type}: {message}"


def _render_operator_refinement_result(result: dict[str, object], *, edit_key: str) -> None:
    mode_key = "operator_rewrite_mode_llm" if result.get("mode") == "llm" else "operator_rewrite_mode_local"
    mode_label = _t(mode_key)
    validation_issues = list(result.get("validation_issues") or [])
    review_result = result.get("review_result")
    review_findings = list(getattr(review_result, "findings", []) or []) if review_result else []
    has_validation_blockers = any(str(getattr(issue, "severity", "")).lower() == "critical" for issue in validation_issues)
    has_review_blockers = any(str(getattr(finding, "severity", "")).lower() in {"critical", "major"} for finding in review_findings)

    st.markdown(_t("operator_rewrite_result"))
    st.caption(_t("operator_rewrite_source", mode=mode_label))
    st.info(str(result.get("proposed_text") or ""))
    if result.get("auto_applied_to_editor", False):
        st.success(_t("operator_rewrite_applied"))
    else:
        st.warning(_t("operator_rewrite_not_auto_applied"))
        if st.button(
            _t("operator_rewrite_apply_result"),
            key=(
                f"operator_rewrite_apply_{edit_key}_"
                f"{hashlib.sha1(str(result.get('proposed_text') or '').encode('utf-8')).hexdigest()[:12]}"
            ),
            icon=":material/low_priority:",
        ):
            st.session_state[edit_key] = str(result.get("proposed_text") or "")
            result["auto_applied_to_editor"] = True
            st.success(_t("operator_rewrite_applied"))

    if result.get("mode") != "llm":
        st.warning(_t("operator_rewrite_local_warning"))

    if review_result:
        st.caption(_t("operator_rewrite_review_verdict", verdict=getattr(review_result, "verdict", "-")))
    else:
        st.caption(_t("operator_rewrite_review_unavailable"))

    if has_validation_blockers or has_review_blockers:
        st.warning(_t("operator_rewrite_check_warning"))
    elif review_result:
        st.success(_t("operator_rewrite_check_ok"))

    if validation_issues:
        st.markdown(_t("operator_rewrite_validation_issues"))
        for issue in validation_issues:
            st.warning(_issue_label(issue))

    if review_findings:
        st.markdown(_t("operator_rewrite_review_findings"))
        for finding in review_findings:
            st.warning(_issue_label(finding))


def _operator_refinement_note(segment_id: str) -> str | None:
    result = st.session_state.get(_operator_refinement_result_key(segment_id))
    if not result:
        return None
    phrase = str(result.get("preferred_phrase") or "").strip()
    if not phrase:
        return None
    return _t("operator_refinement_note", phrase=phrase)


def _localized_severity(severity: str) -> str:
    severity = severity.lower()
    if _ui_language() == "pl":
        return SEVERITY_LABELS_PL.get(severity, severity.upper())
    return severity.upper()


def _localized_issue_type(issue_type: str) -> str:
    if _ui_language() == "pl":
        return ISSUE_TYPE_LABELS_PL.get(issue_type, issue_type.replace("_", " "))
    return issue_type


def _operation_label(operation: object) -> str:
    operation_text = str(operation or "llm")
    if _ui_language() == "pl":
        return OPERATION_LABELS_PL.get(operation_text, operation_text.replace("_", " "))
    return operation_text


def _display_status_label(status: object) -> str:
    status_text = str(status or "unknown")
    if _ui_language() != "pl":
        return status_text

    exact = {
        "created": "utworzono",
        "pdf_extracted": "PDF wyodrębniony",
        "segments_prepared": "segmenty przygotowane",
        "segments_translated": "tłumaczenie zakończone",
        "translation_reviewed": "recenzja zakończona",
        "invariants_validated": "walidacja zakończona",
        "findings_resolved": "uwagi rozstrzygnięte",
        "needs_human_review": "wymagana decyzja operatora",
        "operator_decisions_applied": "decyzje operatora zastosowane",
        "rendering_with_unresolved_warnings": "generowanie z nierozstrzygniętymi uwagami",
        "pdf_rendered": "PDF wygenerowany",
        "completed": "zakończono",
        "completed_with_output_warnings": "zakończono z ostrzeżeniami",
        "missing_output_pdf": "brak PDF wynikowego",
    }
    if status_text in exact:
        return exact[status_text]

    patterns = [
        (r"^translating (\d+)/(\d+)$", r"tłumaczenie \1/\2"),
        (r"^reviewing (\d+)/(\d+)$", r"recenzja \1/\2"),
        (r"^revising (\d+)/(\d+)$", r"poprawki \1/\2"),
        (r"^pipeline translated (\d+)/(\d+), reviewed (\d+)/(\d+)$", r"pipeline: tłumaczenie \1/\2, recenzja \3/\4"),
    ]
    for pattern, replacement in patterns:
        localized = re.sub(pattern, replacement, status_text)
        if localized != status_text:
            return localized

    return status_text.replace("_", " ")


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


def _token_usage_summary(state: dict) -> dict[str, int]:
    usages = [
        getattr(translation, "token_usage", None)
        for translation in state.get("translations", {}).values()
    ]
    usages.extend(
        getattr(result, "token_usage", None)
        for result in state.get("review_results", {}).values()
    )
    usages = [usage for usage in usages if usage is not None]
    return {
        "requests": len(usages),
        "input_tokens": sum(int(getattr(usage, "input_tokens", 0) or 0) for usage in usages),
        "output_tokens": sum(int(getattr(usage, "output_tokens", 0) or 0) for usage in usages),
        "total_tokens": sum(int(getattr(usage, "total_tokens", 0) or 0) for usage in usages),
    }


def _render_token_usage_metrics(state: dict) -> None:
    usage = _token_usage_summary(state)
    inflight = state.get("llm_inflight")
    inflight = inflight if isinstance(inflight, dict) else None
    estimated_input_tokens = _safe_int(inflight.get("estimated_input_tokens")) if inflight else 0
    active_requests = _safe_int(inflight.get("active")) if inflight else 0
    if inflight and active_requests <= 0:
        active_requests = 1
    request_delta = None
    if active_requests == 1:
        request_delta = _t("llm_request_inflight_delta")
    elif active_requests > 1:
        request_delta = _t("llm_requests_inflight_delta", active=active_requests)
    input_delta = (
        _t("estimated_input_delta", tokens=f"{estimated_input_tokens:,}")
        if estimated_input_tokens
        else None
    )
    total_delta = (
        _t("estimated_total_delta", tokens=f"{estimated_input_tokens:,}")
        if estimated_input_tokens
        else None
    )

    st.markdown(f"##### {_t('token_usage_header')}")
    cols = st.columns(4)
    cols[0].metric(
        _t("llm_requests_metric"),
        f"{usage['requests']:,}",
        delta=request_delta,
        delta_color="off",
    )
    cols[1].metric(
        _t("input_tokens_metric"),
        f"{usage['input_tokens']:,}",
        delta=input_delta,
        delta_color="off",
    )
    cols[2].metric(_t("output_tokens_metric"), f"{usage['output_tokens']:,}")
    cols[3].metric(
        _t("total_tokens_metric"),
        f"{usage['total_tokens']:,}",
        delta=total_delta,
        delta_color="off",
    )
    if inflight:
        if inflight.get("operation") == "pipeline":
            st.caption(
                _t(
                    "llm_pipeline_inflight",
                    active=active_requests,
                    segments=_inflight_segments_label(inflight),
                )
            )
        elif active_requests > 1:
            st.caption(
                _t(
                    "llm_inflight_many",
                    active=active_requests,
                    operation=_operation_label(inflight.get("operation", "llm")),
                    provider=inflight.get("provider", "-"),
                    model=inflight.get("model", "-"),
                    segments=_inflight_segments_label(inflight),
                )
            )
        else:
            st.caption(
                _t(
                    "llm_inflight",
                    operation=_operation_label(inflight.get("operation", "llm")),
                    provider=inflight.get("provider", "-"),
                    model=inflight.get("model", "-"),
                    segment_id=inflight.get("segment_id", "-"),
                )
            )
        if estimated_input_tokens:
            st.caption(_t("estimated_token_note"))
        _render_parallel_tasks(state, inflight)
    elif usage["requests"] == 0:
        st.caption(_t("no_token_usage_yet"))


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _config_int(config: object, name: str, default: int) -> int:
    raw_value = config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)
    value = _safe_int(raw_value)
    return value if value > 0 else default


def _session_concurrency(name: str, fallback: int) -> int:
    value = _safe_int(st.session_state.get(name))
    return value if value > 0 else fallback


def _current_parallelism(fallback_config: object | None = None) -> tuple[int, int]:
    translation_fallback = _config_int(fallback_config, "translation_concurrency", 4) if fallback_config else 4
    review_fallback = _config_int(fallback_config, "review_concurrency", 4) if fallback_config else 4
    return (
        _session_concurrency("translation_concurrency", translation_fallback),
        _session_concurrency("review_concurrency", review_fallback),
    )


def _coerce_job_config(config: JobConfig | dict | object) -> JobConfig:
    if isinstance(config, JobConfig):
        return config
    if isinstance(config, dict):
        return JobConfig.model_validate(config)

    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        return JobConfig.model_validate(model_dump())

    return JobConfig.model_validate(vars(config))


def _with_current_parallelism(config: JobConfig | dict | object) -> JobConfig:
    config = _coerce_job_config(config)

    translation_concurrency, review_concurrency = _current_parallelism(config)
    return config.model_copy(
        update={
            "translation_concurrency": translation_concurrency,
            "review_concurrency": review_concurrency,
        }
    )


def _inflight_segments_label(inflight: dict) -> str:
    segments = inflight.get("segments")
    if not isinstance(segments, list) or not segments:
        return str(inflight.get("segment_id", "-"))

    labels = []
    for item in segments[:6]:
        if isinstance(item, dict):
            labels.append(str(item.get("segment_id", "-")))
        else:
            labels.append(str(item))

    remaining = len(segments) - len(labels)
    suffix = f" +{remaining}" if remaining > 0 else ""
    return ", ".join(labels) + suffix


def _render_parallel_tasks(state: dict, inflight: dict) -> None:
    requests = _inflight_requests(inflight)
    if not requests:
        return

    segments_by_id = {
        getattr(segment, "segment_id", ""): segment
        for segment in state.get("segments", [])
    }
    rows = []
    for index, request in enumerate(requests, start=1):
        segment_id = str(request.get("segment_id") or inflight.get("segment_id") or "-")
        segment = segments_by_id.get(segment_id)
        operation = str(request.get("operation") or inflight.get("operation", "llm"))
        provider = str(request.get("provider") or inflight.get("provider", "-"))
        model = str(request.get("model") or inflight.get("model", "-"))
        rows.append(
            {
                _t("parallel_task_col_slot"): index,
                _t("parallel_task_col_status"): _t("parallel_task_status"),
                _t("parallel_task_col_operation"): _operation_label(operation),
                _t("parallel_task_col_segment"): segment_id,
                _t("parallel_task_col_page"): getattr(segment, "page_number", "-") if segment else "-",
                _t("parallel_task_col_provider"): provider,
                _t("parallel_task_col_model"): model,
                _t("parallel_task_col_estimated_input"): _safe_int(request.get("estimated_input_tokens")),
                _t("parallel_task_col_source"): _short_text(getattr(segment, "source_text", ""), 220) if segment else "",
            }
        )

    st.markdown(f"##### :material/account_tree: {_t('parallel_tasks_header')}")
    st.caption(_t("parallel_tasks_caption"))
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=min(360, 72 + 35 * len(rows)),
    )


def _inflight_requests(inflight: dict) -> list[dict]:
    segments = inflight.get("segments")
    if isinstance(segments, list) and segments:
        return [
            item if isinstance(item, dict) else {"segment_id": str(item)}
            for item in segments
        ]
    if inflight.get("segment_id"):
        return [
            {
                "segment_id": inflight.get("segment_id"),
                "estimated_input_tokens": inflight.get("estimated_input_tokens"),
            }
        ]
    return []


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
        message = _progress_message(str(event.get("message") or _t("progress_working")))
        current = event.get("current")
        total = event.get("total")
        segment_id = event.get("segment_id")
        pipeline_progress_label = _pipeline_progress_label(event) if stage == "pipeline" else None
        status_label = message

        if isinstance(current, int) and isinstance(total, int) and total > 0:
            ratio = min(max(current / total, 0.0), 1.0)
            progress_text = (
                f"{message} | {pipeline_progress_label}"
                if pipeline_progress_label
                else f"{message}: {current}/{total}"
            )
            if segment_id:
                progress_text = f"{progress_text} - {segment_id}"
            status_label = progress_text
            progress_bar.progress(ratio, text=progress_text)
            stage_progress = pipeline_progress_label or f"{current}/{total}"
            detail_slot.caption(
                f"{_t('progress_stage')}: `{stage}` | {_t('progress_segment')}: `{segment_id or '-'}` | "
                f"{_t('progress_stage_progress')}: {stage_progress}"
            )
            log_key = (
                (stage, _safe_int(event.get("translations_done")), _safe_int(event.get("reviews_done")))
                if pipeline_progress_label
                else (stage, current // 10)
            )
        else:
            ratio = 1.0 if stage in {"done", "human_review"} else 0.0
            progress_bar.progress(ratio, text=message)
            detail_slot.caption(f"{_t('progress_stage')}: `{stage}`")
            log_key = (stage, message)

        _status_update(status, label=status_label)

        if stage != last_stage["value"]:
            history_slot.write(f"**{message}**")
            last_stage["value"] = stage
            last_log_key["value"] = log_key
        elif log_key != last_log_key["value"] and isinstance(current, int) and isinstance(total, int):
            if pipeline_progress_label:
                history_slot.caption(f"{message}: {pipeline_progress_label}")
            else:
                history_slot.caption(f"{message}: {current}/{total}")
            last_log_key["value"] = log_key

        if live_preview_slot is not None and stage in {"extract", "prepare", "translate", "review", "revise", "pipeline"}:
            preview_state = _load_latest_checkpoint(silent=True)
            if not preview_state or not _checkpoint_matches(preview_state, checkpoint_job_id, checkpoint_source_path):
                return

            translations_count = len(preview_state.get("translations", {}))
            usage = _token_usage_summary(preview_state)
            preview_key = (
                preview_state.get("job_id"),
                preview_state.get("status"),
                len(preview_state.get("segments", [])),
                translations_count,
                len(preview_state.get("review_results", {})),
                _review_findings_count(preview_state),
                usage["total_tokens"],
                usage["requests"],
                preview_state.get("llm_inflight"),
            )
            if preview_key != last_preview_key["value"]:
                last_preview_key["value"] = preview_key
                live_preview_slot.empty()
                with live_preview_slot.container():
                    _render_live_translation_preview(
                        preview_state,
                        title=_t("live_preview"),
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
    translation_concurrency: int,
    review_concurrency: int,
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
        translation_concurrency=translation_concurrency,
        review_concurrency=review_concurrency,
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


def _progress_counts(state: dict) -> dict[str, int]:
    segments_total = len(state.get("segments", []))
    translations_done = len(state.get("translations", {}))
    reviews_done = len(state.get("review_results", {}))
    return {
        "segments_total": segments_total,
        "translations_done": translations_done,
        "reviews_done": reviews_done,
    }


def _progress_percent(done: int, total: int) -> int:
    if total <= 0:
        return 0
    if done >= total:
        return 100
    return max(0, int(done * 100 / total))


def _progress_ratio(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return min(max(done / total, 0.0), 1.0)


def _output_pdf_exists(state: dict) -> bool:
    output_pdf_path = state.get("output_pdf_path")
    return bool(output_pdf_path) and Path(str(output_pdf_path)).exists()


def _ready_for_pdf_generation(state: dict) -> bool:
    if state.get("output_pdf_path") or state.get("unresolved_segments"):
        return False

    counts = _progress_counts(state)
    total = counts["segments_total"]
    if total <= 0:
        return False

    return (
        counts["translations_done"] >= total
        and counts["reviews_done"] >= total
        and state.get("status") not in {"completed", "completed_with_output_warnings", "needs_human_review"}
    )


def _pdf_status_label(state: dict) -> str:
    if _output_pdf_exists(state):
        return _t("pdf_status_download")
    if state.get("output_pdf_path"):
        return _t("pdf_status_missing")
    if state.get("unresolved_segments"):
        return _t("pdf_status_needs_decisions")
    if _ready_for_pdf_generation(state):
        return _t("pdf_status_ready_to_generate")
    if state.get("segments"):
        return _t("pdf_status_progress")
    return _t("pdf_status_waiting")


def _render_status(state: dict) -> None:
    config = state.get("config")
    provider_note = ""
    if config:
        translation_concurrency = _config_int(config, "translation_concurrency", 4)
        review_concurrency = _config_int(config, "review_concurrency", 4)
        provider_note = (
            f" | {_t('source_to_target')}: `{config.source_language} -> {config.target_language}`"
            f" | {_t('translator_provider')}: `{config.translator_provider}`, "
            f"{_t('review_provider')}: `{config.reviewer_provider}`"
            f" | {_t('parallelism_status')}: `{translation_concurrency}/{review_concurrency}`"
    )

    st.subheader(_t("status"))
    st.write(f"`{_display_status_label(state.get('status', 'unknown'))}`{provider_note}")
    if config:
        ui_translation_concurrency, ui_review_concurrency = _current_parallelism(config)
        job_translation_concurrency = _config_int(config, "translation_concurrency", 4)
        job_review_concurrency = _config_int(config, "review_concurrency", 4)
        if (
            ui_translation_concurrency != job_translation_concurrency
            or ui_review_concurrency != job_review_concurrency
        ):
            st.caption(
                _t(
                    "parallelism_change_pending",
                    ui_translation=ui_translation_concurrency,
                    ui_review=ui_review_concurrency,
                    job_translation=job_translation_concurrency,
                    job_review=job_review_concurrency,
                )
            )
    st.caption(
        _t(
            "pipeline_progress_caption",
            translations=len(state.get("translations", {})),
            reviews=len(state.get("review_results", {})),
            segments=len(state.get("segments", [])),
        )
    )
    if config and config.debug:
        log_path = Path(config.output_dir).parent / "logs" / f"{state.get('job_id')}.debug.log"
        st.caption(_t("debug_log", path=log_path))

    segments = state.get("segments", [])
    counts = _progress_counts(state)
    segments_total = counts["segments_total"]
    translations_done = counts["translations_done"]
    reviews_done = counts["reviews_done"]
    translation_percent = _progress_percent(translations_done, segments_total)
    review_percent = _progress_percent(reviews_done, segments_total)
    review_findings = _review_findings_count(state)
    unresolved_segments = state.get("unresolved_segments", [])

    cols = st.columns(6)
    cols[0].metric(_t("segments_metric"), len(segments))
    cols[1].metric(
        _t("translation_progress_metric"),
        f"{translation_percent}%",
        delta=f"{translations_done}/{segments_total or '?'}",
        delta_color="off",
    )
    cols[2].metric(
        _t("review_metric"),
        f"{review_percent}%",
        delta=f"{reviews_done}/{segments_total or '?'}",
        delta_color="off",
    )
    cols[3].metric(_t("validation_metric"), len(state.get("deterministic_issues", [])))
    cols[4].metric(_t("decisions_metric"), len(unresolved_segments))
    cols[5].metric(
        _t("pdf_metric"),
        _pdf_status_label(state),
        delta=_t(
            "pdf_progress_delta",
            translation_percent=translation_percent,
            review_percent=review_percent,
        ),
        delta_color="off",
    )
    st.caption(
        _t(
            "progress_to_pdf",
            translation_percent=translation_percent,
            review_percent=review_percent,
        )
    )
    if segments_total:
        st.progress(
            _progress_ratio(translations_done, segments_total),
            text=_t(
                "translation_progress_bar",
                percent=translation_percent,
                done=translations_done,
                total=segments_total,
            ),
        )
        st.progress(
            _progress_ratio(reviews_done, segments_total),
            text=_t(
                "review_progress_bar",
                percent=review_percent,
                done=reviews_done,
                total=segments_total,
            ),
        )
    st.caption(_t("review_findings_caption", review_findings=review_findings))
    _render_review_findings_table(state)
    _render_token_usage_metrics(state)
    st.caption(
        _t(
            "cache_status",
            job_hits=state.get("translation_memory_hits", 0),
            persistent_hits=state.get("persistent_translation_cache_hits", 0),
            llm_calls=state.get("translation_memory_misses", 0),
        )
    )

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
        _render_token_usage_metrics(state)
        review_findings = _review_findings_count(state)
        st.caption(_t("review_findings_caption", review_findings=review_findings))
        _render_review_findings_table(state)
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
        preview_height = _wrapped_text_area_height(
            last_segment.source_text,
            last_translation.translated_text,
            min_height=160,
            max_height=360,
        )
        left.text_area(
            _t("source_segment_caption", segment_id=last_segment.segment_id, page_number=last_segment.page_number),
            value=last_segment.source_text,
            height=preview_height,
            disabled=True,
            key=_unique_widget_key("live_source_text", state.get("job_id"), done, last_segment.segment_id),
        )
        right.text_area(
            _t("saved_translation_caption"),
            value=last_translation.translated_text,
            height=preview_height,
            disabled=True,
            key=_unique_widget_key(
                "live_translation_text",
                state.get("job_id"),
                done,
                last_segment.segment_id,
                _text_digest(last_translation.translated_text),
            ),
        )

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
            key=_unique_widget_key("live_joined_translation", state.get("job_id"), done, _text_digest(full_text)),
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
    if _ready_for_pdf_generation(state):
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


def _resume_translation(
    state: dict,
    *,
    status_label: str | None = None,
    progress_label: str | None = None,
    done_label: str | None = None,
) -> None:
    config = state.get("config")
    if not config:
        st.session_state["last_error"] = _t("empty_checkpoint")
        return

    config = _with_current_parallelism(config)
    state = {**state, "config": config}

    source_path = Path(state.get("source_pdf_path", ""))
    if not source_path.exists():
        st.session_state["last_error"] = _t("resume_missing_source", path=source_path)
        return

    errors = _validate_provider_configuration(config.translator_provider, config.reviewer_provider)
    if errors:
        st.session_state["last_error"] = "\n".join(errors)
        return

    status = st.status(status_label or _t("resume_status"), expanded=True)
    status.caption(
        _t(
            "resume_parallelism_applied",
            translation=config.translation_concurrency,
            review=config.review_concurrency,
        )
    )
    progress_bar = status.progress(0, text=progress_label or _t("resume_progress"))
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
        _status_update(status, label=done_label or _t("resume_done"), state="complete")


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
            key=_unique_widget_key(
                "checkpoint_joined_translation",
                state.get("job_id"),
                len(translated_pairs),
                _text_digest(full_text),
            ),
        )


def _short_text(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}…"


def _wrapped_text_area_height(*texts: object, min_height: int = 180, max_height: int = 520) -> int:
    visual_lines = 0
    for text in texts:
        raw_lines = str(text or "").splitlines() or [""]
        for line in raw_lines:
            visual_lines += max(1, (len(line) + 84) // 85)
    return min(max_height, max(min_height, 76 + visual_lines * 24))


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


def _unique_widget_key(prefix: str, *parts: object) -> str:
    stable_part = "_".join(str(part) for part in parts)
    return f"{prefix}_{next(WIDGET_KEY_COUNTER)}_{stable_part}"


def _render_human_review(state: dict) -> None:
    unresolved_segments = state.get("unresolved_segments", [])
    if not unresolved_segments:
        return

    config = state.get("config")
    if not config:
        st.error(_t("empty_checkpoint"))
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
        format_func=_localized_severity,
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
        edit_key = f"edit_{segment_id}"
        st.session_state.setdefault(edit_key, translation.translated_text)

        with st.expander(
            _t("review_expander_label", segment_id=segment_id, page_number=segment.page_number),
            expanded=index == 0,
        ):
            left, right = st.columns(2)
            review_preview_height = _wrapped_text_area_height(
                segment.source_text,
                translation.translated_text,
                min_height=220,
                max_height=560,
            )
            left.text_area(
                _t("source"),
                value=segment.source_text,
                height=review_preview_height,
                disabled=True,
                key=f"source_review_{segment_id}",
            )
            right.text_area(
                _t("current_translation"),
                value=translation.translated_text,
                height=review_preview_height,
                disabled=True,
                key=f"current_translation_review_{segment_id}",
            )

            _render_operator_phrase_refinement(
                segment,
                translation,
                config,
                edit_key=edit_key,
                issues=issues,
            )

            edited = st.text_area(
                _t("approved_text"),
                key=edit_key,
                height=_wrapped_text_area_height(
                    st.session_state.get(edit_key, translation.translated_text),
                    min_height=220,
                    max_height=560,
                ),
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

            decisions[segment_id] = {
                "action": action or "edit",
                "text": edited,
                "note": _operator_refinement_note(segment_id),
            }

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


def _render_pdf_generation_action(state: dict) -> None:
    if not _ready_for_pdf_generation(state):
        return

    with st.container(border=True):
        st.success(_t("pdf_ready_to_generate"))
        if st.button(
            _t("generate_pdf"),
            type="primary",
            icon=":material/picture_as_pdf:",
            key=f"generate_pdf_{state.get('job_id', 'current')}",
        ):
            _resume_translation(
                state,
                status_label=_t("generating_pdf"),
                progress_label=_t("rendering_start"),
                done_label=_t("pdf_generated"),
            )
            st.rerun()


def _render_outputs(state: dict) -> None:
    if not state.get("output_pdf_path"):
        return

    output_path = Path(state["output_pdf_path"])
    if not output_path.exists():
        st.error(_t("output_pdf_missing", path=output_path))
        return

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


st.set_page_config(page_title="TechnicAl. PDF Translator -- delta version", layout="wide")
_init_session_state()

openai_ready = _has_env_value("OPENAI_API_KEY")
anthropic_ready = _has_env_value("ANTHROPIC_API_KEY")
loaded_state = st.session_state.get("translation_state")
loaded_config = loaded_state.get("config") if isinstance(loaded_state, dict) else None

with st.sidebar:
    st.markdown("*PuffyClouds*")
    brand_col, version_col = st.columns([0.72, 0.28], vertical_alignment="center")
    brand_col.markdown(
        '<span style="font-size: 3rem; font-weight: 800; line-height: 1;">TechnicAl</span>',
        unsafe_allow_html=True,
    )
    version_col.caption(f"`v{_app_version()}`", text_alignment="right")
    st.caption("Cross-language technical document translation agent.")
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
    st.subheader(_t("parallelism"))
    translation_concurrency = st.slider(
        _t("translation_concurrency"),
        min_value=1,
        max_value=12,
        value=_session_concurrency(
            "translation_concurrency",
            _config_int(loaded_config, "translation_concurrency", 4) if loaded_config else 4,
        ),
        help=_t("concurrency_help"),
        key="translation_concurrency",
    )
    review_concurrency = st.slider(
        _t("review_concurrency"),
        min_value=1,
        max_value=12,
        value=_session_concurrency(
            "review_concurrency",
            _config_int(loaded_config, "review_concurrency", 4) if loaded_config else 4,
        ),
        help=_t("concurrency_help"),
        key="review_concurrency",
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
            translation_concurrency,
            review_concurrency,
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
    _render_pdf_generation_action(state)
    _render_outputs(state)
else:
    st.info(_t("empty_info"))
